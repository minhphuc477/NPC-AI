# Proposal Alignment Evaluation Report

- Run ID: `20260306T075851Z`
- Generated: `2026-03-06T08:00:25.240542+00:00`
- Scenarios: `artifacts\proposal_control_tuning\near_pass_smoke\20260306T075851Z\scenarios.jsonl`
- Scenario count: `2`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_alt`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2542 (0.2542, 0.2542) | 0.3333 (0.3333, 0.3333) | 0.9418 (0.9418, 0.9418) | 0.4165 (0.4165, 0.4165) | n/a |
| proposed_contextual_controlled_alt | 0.3721 (0.2242, 0.5199) | 0.2833 (0.2333, 0.3333) | 0.8750 (0.8148, 0.9352) | 0.4415 (0.3455, 0.5375) | n/a |
| proposed_contextual | 0.1013 (0.0088, 0.1937) | 0.1500 (0.1000, 0.2000) | 0.8577 (0.7486, 0.9667) | 0.2630 (0.1829, 0.3431) | n/a |
| candidate_no_context | 0.0078 (0.0068, 0.0088) | 0.1500 (0.1000, 0.2000) | 0.7486 (0.7486, 0.7486) | 0.1975 (0.1829, 0.2122) | n/a |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0935 | 11.9509 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | naturalness | 0.1090 | 0.1456 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.1250 | nan |
| proposed_vs_candidate_no_context | context_overlap | 0.0199 | 0.7628 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0000 | nan |
| proposed_vs_candidate_no_context | persona_style | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | distinct1 | 0.0455 | 0.0500 |
| proposed_vs_candidate_no_context | length_score | 0.3667 | 3.6667 |
| proposed_vs_candidate_no_context | sentence_score | 0.1750 | 0.2692 |
| proposed_vs_candidate_no_context | overall_quality | 0.0655 | 0.3313 |
| controlled_vs_proposed_raw | context_relevance | 0.1529 | 1.5094 |
| controlled_vs_proposed_raw | persona_consistency | 0.1833 | 1.2222 |
| controlled_vs_proposed_raw | naturalness | 0.0842 | 0.0981 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2083 | 1.6667 |
| controlled_vs_proposed_raw | context_overlap | 0.0235 | 0.5111 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1667 | nan |
| controlled_vs_proposed_raw | persona_style | 0.2500 | 0.3333 |
| controlled_vs_proposed_raw | distinct1 | 0.0000 | 0.0000 |
| controlled_vs_proposed_raw | length_score | 0.3333 | 0.7143 |
| controlled_vs_proposed_raw | sentence_score | 0.1750 | 0.2121 |
| controlled_vs_proposed_raw | overall_quality | 0.1535 | 0.5838 |
| controlled_vs_candidate_no_context | context_relevance | 0.2463 | 31.4986 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1833 | 1.2222 |
| controlled_vs_candidate_no_context | naturalness | 0.1932 | 0.2580 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3333 | nan |
| controlled_vs_candidate_no_context | context_overlap | 0.0434 | 1.6638 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1667 | nan |
| controlled_vs_candidate_no_context | persona_style | 0.2500 | 0.3333 |
| controlled_vs_candidate_no_context | distinct1 | 0.0455 | 0.0500 |
| controlled_vs_candidate_no_context | length_score | 0.7000 | 7.0000 |
| controlled_vs_candidate_no_context | sentence_score | 0.3500 | 0.5385 |
| controlled_vs_candidate_no_context | overall_quality | 0.2190 | 1.1086 |
| controlled_alt_vs_controlled_default | context_relevance | 0.1179 | 0.4639 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0500 | -0.1500 |
| controlled_alt_vs_controlled_default | naturalness | -0.0668 | -0.0709 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.1364 | 0.4091 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0748 | 1.0774 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0000 | 0.0000 |
| controlled_alt_vs_controlled_default | persona_style | -0.2500 | -0.2500 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0438 | -0.0459 |
| controlled_alt_vs_controlled_default | length_score | -0.2333 | -0.2917 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0000 | 0.0000 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0250 | 0.0600 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.2708 | 2.6734 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1333 | 0.8889 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0174 | 0.0203 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.3447 | 2.7576 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0983 | 2.1392 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1667 | nan |
| controlled_alt_vs_proposed_raw | persona_style | 0.0000 | 0.0000 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0438 | -0.0459 |
| controlled_alt_vs_proposed_raw | length_score | 0.1000 | 0.2143 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1750 | 0.2121 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1785 | 0.6788 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.3642 | 46.5738 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1333 | 0.8889 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.1264 | 0.1688 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.4697 | nan |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.1182 | 4.5339 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1667 | nan |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0000 | 0.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0016 | 0.0018 |
| controlled_alt_vs_candidate_no_context | length_score | 0.4667 | 4.6667 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.3500 | 0.5385 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.2440 | 1.2351 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0935 | (0.0000, 0.1869) | 0.2453 | 0.0935 | (0.0000, 0.1869) | 0.2473 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | naturalness | 0.1090 | (0.0000, 0.2180) | 0.2407 | 0.1090 | (0.0000, 0.2180) | 0.2340 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.1250 | (0.0000, 0.2500) | 0.2387 | 0.1250 | (0.0000, 0.2500) | 0.2400 |
| proposed_vs_candidate_no_context | context_overlap | 0.0199 | (0.0000, 0.0398) | 0.2450 | 0.0199 | (0.0000, 0.0398) | 0.2420 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | distinct1 | 0.0455 | (0.0000, 0.0909) | 0.2587 | 0.0455 | (0.0000, 0.0909) | 0.2503 |
| proposed_vs_candidate_no_context | length_score | 0.3667 | (0.0000, 0.7333) | 0.2553 | 0.3667 | (0.0000, 0.7333) | 0.2533 |
| proposed_vs_candidate_no_context | sentence_score | 0.1750 | (0.0000, 0.3500) | 0.2350 | 0.1750 | (0.0000, 0.3500) | 0.2480 |
| proposed_vs_candidate_no_context | overall_quality | 0.0655 | (0.0000, 0.1309) | 0.2497 | 0.0655 | (0.0000, 0.1309) | 0.2493 |
| controlled_vs_proposed_raw | context_relevance | 0.0604 | (0.0604, 0.0604) | 0.0000 | 0.0604 | (0.0604, 0.0604) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1333 | (0.1333, 0.1333) | 0.0000 | 0.1333 | (0.1333, 0.1333) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | -0.0248 | (-0.0248, -0.0248) | 1.0000 | -0.0248 | (-0.0248, -0.0248) | 1.0000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.0833 | (0.0833, 0.0833) | 0.0000 | 0.0833 | (0.0833, 0.0833) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0069 | (0.0069, 0.0069) | 0.0000 | 0.0069 | (0.0069, 0.0069) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1667 | (0.1667, 0.1667) | 0.0000 | 0.1667 | (0.1667, 0.1667) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | distinct1 | -0.0455 | (-0.0455, -0.0455) | 1.0000 | -0.0455 | (-0.0455, -0.0455) | 1.0000 |
| controlled_vs_proposed_raw | length_score | -0.0333 | (-0.0333, -0.0333) | 1.0000 | -0.0333 | (-0.0333, -0.0333) | 1.0000 |
| controlled_vs_proposed_raw | sentence_score | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | overall_quality | 0.0734 | (0.0734, 0.0734) | 0.0000 | 0.0734 | (0.0734, 0.0734) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2473 | (0.2473, 0.2473) | 0.0000 | 0.2473 | (0.2473, 0.2473) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1333 | (0.1333, 0.1333) | 0.0000 | 0.1333 | (0.1333, 0.1333) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.1932 | (0.1932, 0.1932) | 0.0000 | 0.1932 | (0.1932, 0.1932) | 0.0000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3333 | (0.3333, 0.3333) | 0.0000 | 0.3333 | (0.3333, 0.3333) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0467 | (0.0467, 0.0467) | 0.0000 | 0.0467 | (0.0467, 0.0467) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1667 | (0.1667, 0.1667) | 0.0000 | 0.1667 | (0.1667, 0.1667) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | distinct1 | 0.0455 | (0.0455, 0.0455) | 0.0000 | 0.0455 | (0.0455, 0.0455) | 0.0000 |
| controlled_vs_candidate_no_context | length_score | 0.7000 | (0.7000, 0.7000) | 0.0000 | 0.7000 | (0.7000, 0.7000) | 0.0000 |
| controlled_vs_candidate_no_context | sentence_score | 0.3500 | (0.3500, 0.3500) | 0.0000 | 0.3500 | (0.3500, 0.3500) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.2043 | (0.2043, 0.2043) | 0.0000 | 0.2043 | (0.2043, 0.2043) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 0.2657 | (0.2657, 0.2657) | 0.0000 | 0.2657 | (0.2657, 0.2657) | 0.0000 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | naturalness | -0.0066 | (-0.0066, -0.0066) | 1.0000 | -0.0066 | (-0.0066, -0.0066) | 1.0000 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.3333 | (0.3333, 0.3333) | 0.0000 | 0.3333 | (0.3333, 0.3333) | 0.0000 |
| controlled_alt_vs_controlled_default | context_overlap | 0.1080 | (0.1080, 0.1080) | 0.0000 | 0.1080 | (0.1080, 0.1080) | 0.0000 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0498 | (-0.0498, -0.0498) | 1.0000 | -0.0498 | (-0.0498, -0.0498) | 1.0000 |
| controlled_alt_vs_controlled_default | length_score | 0.0667 | (0.0667, 0.0667) | 0.0000 | 0.0667 | (0.0667, 0.0667) | 0.0000 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | overall_quality | 0.1210 | (0.1210, 0.1210) | 0.0000 | 0.1210 | (0.1210, 0.1210) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.2708 | (0.2154, 0.3261) | 0.0000 | 0.2708 | (0.2154, 0.3261) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1333 | (0.1333, 0.1333) | 0.0000 | 0.1333 | (0.1333, 0.1333) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0174 | (-0.0314, 0.0662) | 0.2407 | 0.0174 | (-0.0314, 0.0662) | 0.2617 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.3447 | (0.2727, 0.4167) | 0.0000 | 0.3447 | (0.2727, 0.4167) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0983 | (0.0817, 0.1149) | 0.0000 | 0.0983 | (0.0817, 0.1149) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1667 | (0.1667, 0.1667) | 0.0000 | 0.1667 | (0.1667, 0.1667) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0438 | (-0.0952, 0.0076) | 0.7663 | -0.0438 | (-0.0952, 0.0076) | 0.7603 |
| controlled_alt_vs_proposed_raw | length_score | 0.1000 | (0.0333, 0.1667) | 0.0000 | 0.1000 | (0.0333, 0.1667) | 0.0000 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1750 | (0.0000, 0.3500) | 0.2403 | 0.1750 | (0.0000, 0.3500) | 0.2467 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1785 | (0.1626, 0.1945) | 0.0000 | 0.1785 | (0.1626, 0.1945) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.3642 | (0.2154, 0.5131) | 0.0000 | 0.3642 | (0.2154, 0.5131) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1333 | (0.1333, 0.1333) | 0.0000 | 0.1333 | (0.1333, 0.1333) | 0.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.1264 | (0.0662, 0.1866) | 0.0000 | 0.1264 | (0.0662, 0.1866) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.4697 | (0.2727, 0.6667) | 0.0000 | 0.4697 | (0.2727, 0.6667) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.1182 | (0.0817, 0.1547) | 0.0000 | 0.1182 | (0.0817, 0.1547) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1667 | (0.1667, 0.1667) | 0.0000 | 0.1667 | (0.1667, 0.1667) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0016 | (-0.0043, 0.0076) | 0.2520 | 0.0016 | (-0.0043, 0.0076) | 0.2463 |
| controlled_alt_vs_candidate_no_context | length_score | 0.4667 | (0.1667, 0.7667) | 0.0000 | 0.4667 | (0.1667, 0.7667) | 0.0000 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.3500 | (0.3500, 0.3500) | 0.0000 | 0.3500 | (0.3500, 0.3500) | 0.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.2440 | (0.1626, 0.3254) | 0.0000 | 0.2440 | (0.1626, 0.3254) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 1 | 0 | 1 | 0.7500 | 1.0000 |
| proposed_vs_candidate_no_context | persona_consistency | 0 | 0 | 2 | 0.5000 | nan |
| proposed_vs_candidate_no_context | naturalness | 1 | 0 | 1 | 0.7500 | 1.0000 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 1 | 0 | 1 | 0.7500 | 1.0000 |
| proposed_vs_candidate_no_context | context_overlap | 1 | 0 | 1 | 0.7500 | 1.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0 | 0 | 2 | 0.5000 | nan |
| proposed_vs_candidate_no_context | persona_style | 0 | 0 | 2 | 0.5000 | nan |
| proposed_vs_candidate_no_context | distinct1 | 1 | 0 | 1 | 0.7500 | 1.0000 |
| proposed_vs_candidate_no_context | length_score | 1 | 0 | 1 | 0.7500 | 1.0000 |
| proposed_vs_candidate_no_context | sentence_score | 1 | 0 | 1 | 0.7500 | 1.0000 |
| proposed_vs_candidate_no_context | overall_quality | 1 | 0 | 1 | 0.7500 | 1.0000 |
| controlled_vs_proposed_raw | context_relevance | 1 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_proposed_raw | persona_consistency | 1 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_proposed_raw | naturalness | 0 | 1 | 0 | 0.0000 | 0.0000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 1 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_proposed_raw | context_overlap | 1 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 1 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_proposed_raw | persona_style | 0 | 0 | 1 | 0.5000 | nan |
| controlled_vs_proposed_raw | distinct1 | 0 | 1 | 0 | 0.0000 | 0.0000 |
| controlled_vs_proposed_raw | length_score | 0 | 1 | 0 | 0.0000 | 0.0000 |
| controlled_vs_proposed_raw | sentence_score | 0 | 0 | 1 | 0.5000 | nan |
| controlled_vs_proposed_raw | overall_quality | 1 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_relevance | 1 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 1 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | naturalness | 1 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 1 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 1 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 1 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_style | 0 | 0 | 1 | 0.5000 | nan |
| controlled_vs_candidate_no_context | distinct1 | 1 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | length_score | 1 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | sentence_score | 1 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | overall_quality | 1 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 1 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_controlled_default | persona_consistency | 0 | 0 | 1 | 0.5000 | nan |
| controlled_alt_vs_controlled_default | naturalness | 0 | 1 | 0 | 0.0000 | 0.0000 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 1 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_controlled_default | context_overlap | 1 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0 | 0 | 1 | 0.5000 | nan |
| controlled_alt_vs_controlled_default | persona_style | 0 | 0 | 1 | 0.5000 | nan |
| controlled_alt_vs_controlled_default | distinct1 | 0 | 1 | 0 | 0.0000 | 0.0000 |
| controlled_alt_vs_controlled_default | length_score | 1 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_controlled_default | sentence_score | 0 | 0 | 1 | 0.5000 | nan |
| controlled_alt_vs_controlled_default | overall_quality | 1 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_proposed_raw | context_relevance | 2 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 2 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 1 | 1 | 0 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 2 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 2 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 2 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0 | 0 | 2 | 0.5000 | nan |
| controlled_alt_vs_proposed_raw | distinct1 | 1 | 1 | 0 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | length_score | 2 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_proposed_raw | sentence_score | 1 | 0 | 1 | 0.7500 | 1.0000 |
| controlled_alt_vs_proposed_raw | overall_quality | 2 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 2 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 2 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 2 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 2 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 2 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 2 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0 | 0 | 2 | 0.5000 | nan |
| controlled_alt_vs_candidate_no_context | distinct1 | 1 | 1 | 0 | 0.5000 | 0.5000 |
| controlled_alt_vs_candidate_no_context | length_score | 2 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | sentence_score | 2 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 2 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.5000 | 0.5000 | 0.0000 | 0.5000 | 0.5000 |
| proposed_contextual_controlled_alt | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.5000 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `2`
- Unique template signatures: `2`
- Template signature ratio: `1.0000`
- Effective sample size by source clustering: `2.00`
- Effective sample size by template-signature clustering: `2.00`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
# Proposal Alignment Evaluation Report

- Run ID: `20260306T084340Z`
- Generated: `2026-03-06T08:48:40.194667+00:00`
- Scenarios: `artifacts\proposal_control_tuning\architecture_compare_smoke3\20260306T084340Z\hybrid_balanced\seed_31\attempt_1\20260306T084340Z\scenarios.jsonl`
- Scenario count: `4`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_alt`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2883 (0.1882, 0.4238) | 0.2995 (0.2133, 0.4012) | 0.9051 (0.8666, 0.9436) | 0.4200 (0.3867, 0.4650) | n/a |
| proposed_contextual_controlled_alt | 0.2719 (0.2210, 0.3228) | 0.2715 (0.2446, 0.3007) | 0.9097 (0.8898, 0.9346) | 0.4005 (0.3751, 0.4311) | n/a |
| proposed_contextual | 0.0091 (0.0091, 0.0091) | 0.1000 (0.1000, 0.1000) | 0.7486 (0.7486, 0.7486) | 0.1830 (0.1830, 0.1830) | n/a |
| candidate_no_context | 0.0064 (0.0048, 0.0079) | 0.1000 (0.0000, 0.2000) | 0.8586 (0.7486, 0.9686) | 0.2040 (0.1526, 0.2555) | n/a |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0027 | 0.4279 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | naturalness | -0.1100 | -0.1281 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0000 | nan |
| proposed_vs_candidate_no_context | context_overlap | 0.0091 | 0.4279 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0000 | nan |
| proposed_vs_candidate_no_context | persona_style | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | distinct1 | -0.0312 | -0.0331 |
| proposed_vs_candidate_no_context | length_score | -0.4000 | -0.8000 |
| proposed_vs_candidate_no_context | sentence_score | -0.1750 | -0.2121 |
| proposed_vs_candidate_no_context | overall_quality | -0.0210 | -0.1030 |
| controlled_vs_proposed_raw | context_relevance | 0.2792 | 30.7164 |
| controlled_vs_proposed_raw | persona_consistency | 0.1995 | 1.9952 |
| controlled_vs_proposed_raw | naturalness | 0.1564 | 0.2090 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.3636 | nan |
| controlled_vs_proposed_raw | context_overlap | 0.0823 | 2.7164 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2702 | nan |
| controlled_vs_proposed_raw | persona_style | -0.0833 | -0.1667 |
| controlled_vs_proposed_raw | distinct1 | 0.0255 | 0.0280 |
| controlled_vs_proposed_raw | length_score | 0.6000 | 6.0000 |
| controlled_vs_proposed_raw | sentence_score | 0.2625 | 0.4038 |
| controlled_vs_proposed_raw | overall_quality | 0.2370 | 1.2950 |
| controlled_vs_candidate_no_context | context_relevance | 0.2820 | 44.2872 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1995 | 1.9952 |
| controlled_vs_candidate_no_context | naturalness | 0.0465 | 0.0541 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3636 | nan |
| controlled_vs_candidate_no_context | context_overlap | 0.0914 | 4.3066 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2702 | nan |
| controlled_vs_candidate_no_context | persona_style | -0.0833 | -0.1667 |
| controlled_vs_candidate_no_context | distinct1 | -0.0057 | -0.0060 |
| controlled_vs_candidate_no_context | length_score | 0.2000 | 0.4000 |
| controlled_vs_candidate_no_context | sentence_score | 0.0875 | 0.1061 |
| controlled_vs_candidate_no_context | overall_quality | 0.2160 | 1.0586 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0165 | -0.0571 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0280 | -0.0934 |
| controlled_alt_vs_controlled_default | naturalness | 0.0046 | 0.0051 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0227 | -0.0625 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0019 | -0.0167 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0714 | -0.2643 |
| controlled_alt_vs_controlled_default | persona_style | 0.1458 | 0.3500 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0292 | 0.0313 |
| controlled_alt_vs_controlled_default | length_score | 0.0083 | 0.0119 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0875 | -0.0959 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0195 | -0.0464 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.2628 | 28.9042 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1715 | 1.7155 |
| controlled_alt_vs_proposed_raw | naturalness | 0.1611 | 0.2151 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.3409 | nan |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0804 | 2.6542 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1988 | nan |
| controlled_alt_vs_proposed_raw | persona_style | 0.0625 | 0.1250 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0547 | 0.0602 |
| controlled_alt_vs_proposed_raw | length_score | 0.6083 | 6.0833 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1750 | 0.2692 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.2175 | 1.1884 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2655 | 41.6996 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1715 | 1.7155 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0511 | 0.0595 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3409 | nan |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0895 | 4.2178 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1988 | nan |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0625 | 0.1250 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0236 | 0.0251 |
| controlled_alt_vs_candidate_no_context | length_score | 0.2083 | 0.4167 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0000 | 0.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1965 | 0.9630 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| controlled_vs_proposed_raw | context_relevance | 0.2755 | (0.2755, 0.2755) | 0.0000 | 0.2755 | (0.2755, 0.2755) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1333 | (0.1333, 0.1333) | 0.0000 | 0.1333 | (0.1333, 0.1333) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.1938 | (0.1938, 0.1938) | 0.0000 | 0.1938 | (0.1938, 0.1938) | 0.0000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.3636 | (0.3636, 0.3636) | 0.0000 | 0.3636 | (0.3636, 0.3636) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0697 | (0.0697, 0.0697) | 0.0000 | 0.0697 | (0.0697, 0.0697) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1667 | (0.1667, 0.1667) | 0.0000 | 0.1667 | (0.1667, 0.1667) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | distinct1 | 0.0303 | (0.0303, 0.0303) | 0.0000 | 0.0303 | (0.0303, 0.0303) | 0.0000 |
| controlled_vs_proposed_raw | length_score | 0.7333 | (0.7333, 0.7333) | 0.0000 | 0.7333 | (0.7333, 0.7333) | 0.0000 |
| controlled_vs_proposed_raw | sentence_score | 0.3500 | (0.3500, 0.3500) | 0.0000 | 0.3500 | (0.3500, 0.3500) | 0.0000 |
| controlled_vs_proposed_raw | overall_quality | 0.2170 | (0.2170, 0.2170) | 0.0000 | 0.2170 | (0.2170, 0.2170) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.3500 | (0.2167, 0.4833) | 0.0000 | 0.3500 | (0.2167, 0.4833) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1538 | (0.1143, 0.1933) | 0.0000 | 0.1538 | (0.1143, 0.1933) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0448 | (-0.1065, 0.1961) | 0.2383 | 0.0448 | (-0.1065, 0.1961) | 0.2683 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.4545 | (0.2727, 0.6364) | 0.0000 | 0.4545 | (0.2727, 0.6364) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.1061 | (0.0859, 0.1262) | 0.0000 | 0.1061 | (0.0859, 0.1262) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1714 | (0.1429, 0.2000) | 0.0000 | 0.1714 | (0.1429, 0.2000) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0833 | (0.0000, 0.1667) | 0.2557 | 0.0833 | (0.0000, 0.1667) | 0.2513 |
| controlled_vs_candidate_no_context | distinct1 | -0.0130 | (-0.0455, 0.0195) | 0.7420 | -0.0130 | (-0.0455, 0.0195) | 0.7507 |
| controlled_vs_candidate_no_context | length_score | 0.2500 | (-0.2667, 0.7667) | 0.2697 | 0.2500 | (-0.2667, 0.7667) | 0.2417 |
| controlled_vs_candidate_no_context | sentence_score | 0.0000 | (-0.3500, 0.3500) | 0.7570 | 0.0000 | (-0.3500, 0.3500) | 0.7493 |
| controlled_vs_candidate_no_context | overall_quality | 0.2278 | (0.1214, 0.3341) | 0.0000 | 0.2278 | (0.1214, 0.3341) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0165 | (-0.2012, 0.1481) | 0.5810 | -0.0165 | (-0.2012, 0.1481) | 0.5740 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0280 | (-0.1339, 0.0500) | 0.7250 | -0.0280 | (-0.1339, 0.0500) | 0.7207 |
| controlled_alt_vs_controlled_default | naturalness | 0.0046 | (-0.0460, 0.0564) | 0.4487 | 0.0046 | (-0.0460, 0.0564) | 0.4373 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0227 | (-0.2727, 0.2045) | 0.6203 | -0.0227 | (-0.2727, 0.2045) | 0.6453 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0019 | (-0.0350, 0.0234) | 0.5533 | -0.0019 | (-0.0350, 0.0234) | 0.5660 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0714 | (-0.2143, 0.0000) | 1.0000 | -0.0714 | (-0.2143, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | persona_style | 0.1458 | (0.0000, 0.2917) | 0.0590 | 0.1458 | (0.0000, 0.2917) | 0.0530 |
| controlled_alt_vs_controlled_default | distinct1 | 0.0292 | (0.0046, 0.0538) | 0.0020 | 0.0292 | (0.0046, 0.0538) | 0.0060 |
| controlled_alt_vs_controlled_default | length_score | 0.0083 | (-0.2167, 0.2333) | 0.4347 | 0.0083 | (-0.2167, 0.2333) | 0.4480 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0875 | (-0.2625, 0.0000) | 1.0000 | -0.0875 | (-0.2625, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0195 | (-0.0848, 0.0221) | 0.6897 | -0.0195 | (-0.0848, 0.0221) | 0.6863 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.2830 | (0.2830, 0.2830) | 0.0000 | 0.2830 | (0.2830, 0.2830) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1333 | (0.1333, 0.1333) | 0.0000 | 0.1333 | (0.1333, 0.1333) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.1630 | (0.1630, 0.1630) | 0.0000 | 0.1630 | (0.1630, 0.1630) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.3636 | (0.3636, 0.3636) | 0.0000 | 0.3636 | (0.3636, 0.3636) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0947 | (0.0947, 0.0947) | 0.0000 | 0.0947 | (0.0947, 0.0947) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1667 | (0.1667, 0.1667) | 0.0000 | 0.1667 | (0.1667, 0.1667) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0909 | (0.0909, 0.0909) | 0.0000 | 0.0909 | (0.0909, 0.0909) | 0.0000 |
| controlled_alt_vs_proposed_raw | length_score | 0.6333 | (0.6333, 0.6333) | 0.0000 | 0.6333 | (0.6333, 0.6333) | 0.0000 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.2136 | (0.2136, 0.2136) | 0.0000 | 0.2136 | (0.2136, 0.2136) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2146 | (0.2126, 0.2166) | 0.0000 | 0.2146 | (0.2126, 0.2166) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1871 | (0.1143, 0.2600) | 0.0000 | 0.1871 | (0.1143, 0.2600) | 0.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0570 | (-0.0210, 0.1350) | 0.2523 | 0.0570 | (-0.0210, 0.1350) | 0.2507 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.2727 | (0.2727, 0.2727) | 0.0000 | 0.2727 | (0.2727, 0.2727) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0789 | (0.0723, 0.0856) | 0.0000 | 0.0789 | (0.0723, 0.0856) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1714 | (0.1429, 0.2000) | 0.0000 | 0.1714 | (0.1429, 0.2000) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.2500 | (0.0000, 0.5000) | 0.2630 | 0.2500 | (0.0000, 0.5000) | 0.2430 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0174 | (0.0015, 0.0332) | 0.0000 | 0.0174 | (0.0015, 0.0332) | 0.0000 |
| controlled_alt_vs_candidate_no_context | length_score | 0.2500 | (0.0667, 0.4333) | 0.0000 | 0.2500 | (0.0667, 0.4333) | 0.0000 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0000 | (-0.3500, 0.3500) | 0.7467 | 0.0000 | (-0.3500, 0.3500) | 0.7440 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1769 | (0.1384, 0.2153) | 0.0000 | 0.1769 | (0.1384, 0.2153) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| controlled_vs_proposed_raw | context_relevance | 1 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_proposed_raw | persona_consistency | 1 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_proposed_raw | naturalness | 1 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 1 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_proposed_raw | context_overlap | 1 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 1 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_proposed_raw | persona_style | 0 | 0 | 1 | 0.5000 | nan |
| controlled_vs_proposed_raw | distinct1 | 1 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_proposed_raw | length_score | 1 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_proposed_raw | sentence_score | 1 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_proposed_raw | overall_quality | 1 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_relevance | 2 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 2 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | naturalness | 1 | 1 | 0 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 2 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 2 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 2 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_style | 1 | 0 | 1 | 0.7500 | 1.0000 |
| controlled_vs_candidate_no_context | distinct1 | 1 | 1 | 0 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | length_score | 1 | 1 | 0 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | sentence_score | 1 | 1 | 0 | 0.5000 | 0.5000 |
| controlled_vs_candidate_no_context | overall_quality | 2 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 2 | 2 | 0 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | persona_consistency | 1 | 1 | 2 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | naturalness | 2 | 2 | 0 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 1 | 1 | 2 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | context_overlap | 2 | 2 | 0 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0 | 1 | 3 | 0.3750 | 0.0000 |
| controlled_alt_vs_controlled_default | persona_style | 2 | 0 | 2 | 0.7500 | 1.0000 |
| controlled_alt_vs_controlled_default | distinct1 | 3 | 1 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_controlled_default | length_score | 2 | 2 | 0 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | sentence_score | 0 | 1 | 3 | 0.3750 | 0.0000 |
| controlled_alt_vs_controlled_default | overall_quality | 2 | 2 | 0 | 0.5000 | 0.5000 |
| controlled_alt_vs_proposed_raw | context_relevance | 1 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 1 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 1 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 1 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 1 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 1 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0 | 0 | 1 | 0.5000 | nan |
| controlled_alt_vs_proposed_raw | distinct1 | 1 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_proposed_raw | length_score | 1 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_proposed_raw | sentence_score | 0 | 0 | 1 | 0.5000 | nan |
| controlled_alt_vs_proposed_raw | overall_quality | 1 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 2 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 2 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 1 | 1 | 0 | 0.5000 | 0.5000 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 2 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 2 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 2 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 1 | 0 | 1 | 0.7500 | 1.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | 2 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | length_score | 2 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | sentence_score | 1 | 1 | 0 | 0.5000 | 0.5000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 2 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.2500 | 0.2500 | 0.7500 |
| proposed_contextual_controlled_alt | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.5000 |
| proposed_contextual | 0.7500 | 0.7500 | 1.0000 | 0.0000 | 0.0000 |
| candidate_no_context | 0.5000 | 0.5000 | 0.7500 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `4`
- Unique template signatures: `4`
- Template signature ratio: `1.0000`
- Effective sample size by source clustering: `4.00`
- Effective sample size by template-signature clustering: `4.00`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
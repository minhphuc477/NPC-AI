# Proposal Alignment Evaluation Report

- Run ID: `20260306T084943Z`
- Generated: `2026-03-06T08:54:27.344325+00:00`
- Scenarios: `artifacts\proposal_control_tuning\architecture_compare_smoke3\20260306T084340Z\blend_balanced\seed_31\attempt_1\20260306T084943Z\scenarios.jsonl`
- Scenario count: `4`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_alt`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.1783 (0.1400, 0.2166) | 0.3370 (0.2117, 0.4589) | 0.9267 (0.8565, 0.9756) | 0.3875 (0.3554, 0.4197) | n/a |
| proposed_contextual_controlled_alt | 0.2847 (0.1409, 0.4284) | 0.2912 (0.1967, 0.4012) | 0.9315 (0.8966, 0.9655) | 0.4214 (0.3291, 0.5138) | n/a |
| proposed_contextual | 0.0079 (0.0079, 0.0079) | 0.0000 (0.0000, 0.0000) | 0.7486 (0.7486, 0.7486) | 0.1526 (0.1526, 0.1526) | n/a |
| candidate_no_context | 0.0079 (0.0058, 0.0095) | 0.2018 (0.0500, 0.4054) | 0.8597 (0.7486, 0.9708) | 0.2439 (0.1678, 0.3325) | n/a |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | -0.0000 | -0.0053 |
| proposed_vs_candidate_no_context | persona_consistency | -0.2018 | -1.0000 |
| proposed_vs_candidate_no_context | naturalness | -0.1111 | -0.1292 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0000 | nan |
| proposed_vs_candidate_no_context | context_overlap | -0.0001 | -0.0053 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.1429 | -1.0000 |
| proposed_vs_candidate_no_context | persona_style | -0.4375 | -1.0000 |
| proposed_vs_candidate_no_context | distinct1 | -0.0381 | -0.0402 |
| proposed_vs_candidate_no_context | length_score | -0.3917 | -0.7966 |
| proposed_vs_candidate_no_context | sentence_score | -0.1750 | -0.2121 |
| proposed_vs_candidate_no_context | overall_quality | -0.0913 | -0.3745 |
| controlled_vs_proposed_raw | context_relevance | 0.1704 | 21.5883 |
| controlled_vs_proposed_raw | persona_consistency | 0.3370 | nan |
| controlled_vs_proposed_raw | naturalness | 0.1780 | 0.2378 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2273 | nan |
| controlled_vs_proposed_raw | context_overlap | 0.0378 | 1.4368 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.3119 | nan |
| controlled_vs_proposed_raw | persona_style | 0.4375 | nan |
| controlled_vs_proposed_raw | distinct1 | 0.0451 | 0.0496 |
| controlled_vs_proposed_raw | length_score | 0.6250 | 6.2500 |
| controlled_vs_proposed_raw | sentence_score | 0.3500 | 0.5385 |
| controlled_vs_proposed_raw | overall_quality | 0.2350 | 1.5400 |
| controlled_vs_candidate_no_context | context_relevance | 0.1704 | 21.4684 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1352 | 0.6702 |
| controlled_vs_candidate_no_context | naturalness | 0.0669 | 0.0779 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.2273 | nan |
| controlled_vs_candidate_no_context | context_overlap | 0.0377 | 1.4239 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1690 | 1.1833 |
| controlled_vs_candidate_no_context | persona_style | 0.0000 | 0.0000 |
| controlled_vs_candidate_no_context | distinct1 | 0.0070 | 0.0073 |
| controlled_vs_candidate_no_context | length_score | 0.2333 | 0.4746 |
| controlled_vs_candidate_no_context | sentence_score | 0.1750 | 0.2121 |
| controlled_vs_candidate_no_context | overall_quality | 0.1436 | 0.5889 |
| controlled_alt_vs_controlled_default | context_relevance | 0.1064 | 0.5964 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0458 | -0.1360 |
| controlled_alt_vs_controlled_default | naturalness | 0.0048 | 0.0052 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.1364 | 0.6000 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0363 | 0.5666 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0417 | -0.1336 |
| controlled_alt_vs_controlled_default | persona_style | -0.0625 | -0.1429 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0036 | -0.0038 |
| controlled_alt_vs_controlled_default | length_score | 0.0750 | 0.1034 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0875 | -0.0875 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0339 | 0.0875 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.2768 | 35.0599 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.2912 | nan |
| controlled_alt_vs_proposed_raw | naturalness | 0.1828 | 0.2442 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.3636 | nan |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0741 | 2.8175 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2702 | nan |
| controlled_alt_vs_proposed_raw | persona_style | 0.3750 | nan |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0415 | 0.0456 |
| controlled_alt_vs_proposed_raw | length_score | 0.7000 | 7.0000 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.2625 | 0.4038 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.2689 | 1.7623 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2767 | 34.8684 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.0894 | 0.4431 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0718 | 0.0835 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3636 | nan |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0740 | 2.7972 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1274 | 0.8917 |
| controlled_alt_vs_candidate_no_context | persona_style | -0.0625 | -0.1429 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0034 | 0.0036 |
| controlled_alt_vs_candidate_no_context | length_score | 0.3083 | 0.6271 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0875 | 0.1061 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1775 | 0.7279 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | naturalness | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | distinct1 | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | length_score | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | sentence_score | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | overall_quality | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | context_relevance | 0.2044 | (0.2044, 0.2044) | 0.0000 | 0.2044 | (0.2044, 0.2044) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1600 | (0.1600, 0.1600) | 0.0000 | 0.1600 | (0.1600, 0.1600) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.1938 | (0.1938, 0.1938) | 0.0000 | 0.1938 | (0.1938, 0.1938) | 0.0000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2727 | (0.2727, 0.2727) | 0.0000 | 0.2727 | (0.2727, 0.2727) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0451 | (0.0451, 0.0451) | 0.0000 | 0.0451 | (0.0451, 0.0451) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2000 | (0.2000, 0.2000) | 0.0000 | 0.2000 | (0.2000, 0.2000) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_proposed_raw | distinct1 | 0.0303 | (0.0303, 0.0303) | 0.0000 | 0.0303 | (0.0303, 0.0303) | 0.0000 |
| controlled_vs_proposed_raw | length_score | 0.7333 | (0.7333, 0.7333) | 0.0000 | 0.7333 | (0.7333, 0.7333) | 0.0000 |
| controlled_vs_proposed_raw | sentence_score | 0.3500 | (0.3500, 0.3500) | 0.0000 | 0.3500 | (0.3500, 0.3500) | 0.0000 |
| controlled_vs_proposed_raw | overall_quality | 0.1942 | (0.1942, 0.1942) | 0.0000 | 0.1942 | (0.1942, 0.1942) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.1704 | (0.1327, 0.2081) | 0.0000 | 0.1704 | (0.1327, 0.2081) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1352 | (0.0400, 0.2286) | 0.0053 | 0.1352 | (0.0400, 0.2286) | 0.0023 |
| controlled_vs_candidate_no_context | naturalness | 0.0669 | (-0.0653, 0.1992) | 0.1867 | 0.0669 | (-0.0653, 0.1992) | 0.1757 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.2273 | (0.1818, 0.2727) | 0.0000 | 0.2273 | (0.1818, 0.2727) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0377 | (0.0179, 0.0574) | 0.0000 | 0.0377 | (0.0179, 0.0574) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1690 | (0.0500, 0.3000) | 0.0033 | 0.1690 | (0.0500, 0.2857) | 0.0013 |
| controlled_vs_candidate_no_context | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | distinct1 | 0.0070 | (-0.0625, 0.0682) | 0.4617 | 0.0070 | (-0.0625, 0.0682) | 0.4397 |
| controlled_vs_candidate_no_context | length_score | 0.2333 | (-0.2333, 0.7000) | 0.1797 | 0.2333 | (-0.2333, 0.7000) | 0.1840 |
| controlled_vs_candidate_no_context | sentence_score | 0.1750 | (0.0000, 0.3500) | 0.0647 | 0.1750 | (0.0000, 0.3500) | 0.0637 |
| controlled_vs_candidate_no_context | overall_quality | 0.1436 | (0.0707, 0.2166) | 0.0000 | 0.1436 | (0.0529, 0.2166) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 0.1064 | (-0.0336, 0.2463) | 0.0630 | 0.1064 | (-0.0336, 0.2463) | 0.0643 |
| controlled_alt_vs_controlled_default | persona_consistency | -0.0458 | (-0.1000, 0.0000) | 1.0000 | -0.0458 | (-0.1000, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | naturalness | 0.0048 | (-0.0868, 0.1090) | 0.4840 | 0.0048 | (-0.0790, 0.1090) | 0.4660 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.1364 | (-0.0455, 0.3182) | 0.0820 | 0.1364 | (-0.0455, 0.3182) | 0.0803 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0363 | (-0.0061, 0.0787) | 0.0623 | 0.0363 | (-0.0061, 0.0787) | 0.0577 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | -0.0417 | (-0.1250, 0.0000) | 1.0000 | -0.0417 | (-0.1250, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | persona_style | -0.0625 | (-0.1875, 0.0000) | 1.0000 | -0.0625 | (-0.1875, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0036 | (-0.0686, 0.0778) | 0.5607 | -0.0036 | (-0.0686, 0.0778) | 0.5897 |
| controlled_alt_vs_controlled_default | length_score | 0.0750 | (-0.2417, 0.4000) | 0.3517 | 0.0750 | (-0.2417, 0.4000) | 0.3633 |
| controlled_alt_vs_controlled_default | sentence_score | -0.0875 | (-0.2625, 0.0000) | 1.0000 | -0.0875 | (-0.2625, 0.0000) | 1.0000 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0339 | (-0.0291, 0.1103) | 0.2237 | 0.0339 | (-0.0291, 0.1103) | 0.2040 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.2044 | (0.2044, 0.2044) | 0.0000 | 0.2044 | (0.2044, 0.2044) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1600 | (0.1600, 0.1600) | 0.0000 | 0.1600 | (0.1600, 0.1600) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.1938 | (0.1938, 0.1938) | 0.0000 | 0.1938 | (0.1938, 0.1938) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2727 | (0.2727, 0.2727) | 0.0000 | 0.2727 | (0.2727, 0.2727) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0451 | (0.0451, 0.0451) | 0.0000 | 0.0451 | (0.0451, 0.0451) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2000 | (0.2000, 0.2000) | 0.0000 | 0.2000 | (0.2000, 0.2000) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0000 | (0.0000, 0.0000) | 1.0000 | 0.0000 | (0.0000, 0.0000) | 1.0000 |
| controlled_alt_vs_proposed_raw | distinct1 | 0.0303 | (0.0303, 0.0303) | 0.0000 | 0.0303 | (0.0303, 0.0303) | 0.0000 |
| controlled_alt_vs_proposed_raw | length_score | 0.7333 | (0.7333, 0.7333) | 0.0000 | 0.7333 | (0.7333, 0.7333) | 0.0000 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.3500 | (0.3500, 0.3500) | 0.0000 | 0.3500 | (0.3500, 0.3500) | 0.0000 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1942 | (0.1942, 0.1942) | 0.0000 | 0.1942 | (0.1942, 0.1942) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2767 | (0.1346, 0.4189) | 0.0000 | 0.2767 | (0.1346, 0.4189) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.0894 | (-0.0042, 0.1486) | 0.0337 | 0.0894 | (-0.0042, 0.1486) | 0.0360 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0718 | (-0.0401, 0.1836) | 0.1357 | 0.0718 | (-0.0401, 0.1836) | 0.1310 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3636 | (0.1818, 0.5455) | 0.0000 | 0.3636 | (0.1818, 0.5455) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0740 | (0.0244, 0.1236) | 0.0000 | 0.0740 | (0.0244, 0.1236) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1274 | (0.0357, 0.1857) | 0.0050 | 0.1274 | (0.0417, 0.1857) | 0.0023 |
| controlled_alt_vs_candidate_no_context | persona_style | -0.0625 | (-0.1875, 0.0000) | 1.0000 | -0.0625 | (-0.1875, 0.0000) | 1.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0034 | (-0.0526, 0.0327) | 0.2613 | 0.0034 | (-0.0524, 0.0327) | 0.2600 |
| controlled_alt_vs_candidate_no_context | length_score | 0.3083 | (-0.1500, 0.7667) | 0.0943 | 0.3083 | (-0.1500, 0.7667) | 0.0933 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.0875 | (0.0000, 0.2625) | 0.3093 | 0.0875 | (0.0000, 0.2625) | 0.3053 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1775 | (0.0888, 0.2556) | 0.0000 | 0.1775 | (0.0888, 0.2556) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0 | 0 | 1 | 0.5000 | nan |
| proposed_vs_candidate_no_context | persona_consistency | 0 | 0 | 1 | 0.5000 | nan |
| proposed_vs_candidate_no_context | naturalness | 0 | 0 | 1 | 0.5000 | nan |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0 | 0 | 1 | 0.5000 | nan |
| proposed_vs_candidate_no_context | context_overlap | 0 | 0 | 1 | 0.5000 | nan |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0 | 0 | 1 | 0.5000 | nan |
| proposed_vs_candidate_no_context | persona_style | 0 | 0 | 1 | 0.5000 | nan |
| proposed_vs_candidate_no_context | distinct1 | 0 | 0 | 1 | 0.5000 | nan |
| proposed_vs_candidate_no_context | length_score | 0 | 0 | 1 | 0.5000 | nan |
| proposed_vs_candidate_no_context | sentence_score | 0 | 0 | 1 | 0.5000 | nan |
| proposed_vs_candidate_no_context | overall_quality | 0 | 0 | 1 | 0.5000 | nan |
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
| controlled_vs_candidate_no_context | context_relevance | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 3 | 0 | 1 | 0.8750 | 1.0000 |
| controlled_vs_candidate_no_context | naturalness | 3 | 1 | 0 | 0.7500 | 0.7500 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 3 | 0 | 1 | 0.8750 | 1.0000 |
| controlled_vs_candidate_no_context | persona_style | 0 | 0 | 4 | 0.5000 | nan |
| controlled_vs_candidate_no_context | distinct1 | 2 | 1 | 1 | 0.6250 | 0.6667 |
| controlled_vs_candidate_no_context | length_score | 3 | 1 | 0 | 0.7500 | 0.7500 |
| controlled_vs_candidate_no_context | sentence_score | 2 | 0 | 2 | 0.7500 | 1.0000 |
| controlled_vs_candidate_no_context | overall_quality | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 2 | 1 | 1 | 0.6250 | 0.6667 |
| controlled_alt_vs_controlled_default | persona_consistency | 0 | 2 | 2 | 0.2500 | 0.0000 |
| controlled_alt_vs_controlled_default | naturalness | 1 | 2 | 1 | 0.3750 | 0.3333 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 2 | 1 | 1 | 0.6250 | 0.6667 |
| controlled_alt_vs_controlled_default | context_overlap | 2 | 1 | 1 | 0.6250 | 0.6667 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0 | 1 | 3 | 0.3750 | 0.0000 |
| controlled_alt_vs_controlled_default | persona_style | 0 | 1 | 3 | 0.3750 | 0.0000 |
| controlled_alt_vs_controlled_default | distinct1 | 1 | 2 | 1 | 0.3750 | 0.3333 |
| controlled_alt_vs_controlled_default | length_score | 2 | 1 | 1 | 0.6250 | 0.6667 |
| controlled_alt_vs_controlled_default | sentence_score | 0 | 1 | 3 | 0.3750 | 0.0000 |
| controlled_alt_vs_controlled_default | overall_quality | 2 | 1 | 1 | 0.6250 | 0.6667 |
| controlled_alt_vs_proposed_raw | context_relevance | 1 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 1 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 1 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 1 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 1 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 1 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0 | 0 | 1 | 0.5000 | nan |
| controlled_alt_vs_proposed_raw | distinct1 | 1 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_proposed_raw | length_score | 1 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_proposed_raw | sentence_score | 1 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_proposed_raw | overall_quality | 1 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 3 | 1 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_candidate_no_context | naturalness | 3 | 1 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 4 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 3 | 0 | 1 | 0.8750 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0 | 1 | 3 | 0.3750 | 0.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | 3 | 1 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_candidate_no_context | length_score | 3 | 1 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_candidate_no_context | sentence_score | 1 | 0 | 3 | 0.6250 | 1.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 4 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.2500 | 0.5000 | 0.5000 |
| proposed_contextual_controlled_alt | 0.0000 | 0.0000 | 0.2500 | 0.5000 | 0.5000 |
| proposed_contextual | 0.7500 | 0.7500 | 1.0000 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `4`
- Unique template signatures: `4`
- Template signature ratio: `1.0000`
- Effective sample size by source clustering: `4.00`
- Effective sample size by template-signature clustering: `4.00`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
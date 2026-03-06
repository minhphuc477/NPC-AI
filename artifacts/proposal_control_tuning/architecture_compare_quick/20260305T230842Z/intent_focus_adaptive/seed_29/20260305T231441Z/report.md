# Proposal Alignment Evaluation Report

- Run ID: `20260305T231441Z`
- Generated: `2026-03-05T23:17:05.961926+00:00`
- Scenarios: `artifacts\proposal_control_tuning\architecture_compare_quick\20260305T230842Z\intent_focus_adaptive\seed_29\20260305T231441Z\scenarios.jsonl`
- Scenario count: `12`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_alt`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off
- `baseline_no_context`: model `phi3:mini`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.3136 (0.2552, 0.3850) | 0.2782 (0.2416, 0.3167) | 0.8949 (0.8539, 0.9292) | 0.4194 (0.3934, 0.4461) | n/a |
| proposed_contextual_controlled_alt | 0.2552 (0.2155, 0.2898) | 0.3239 (0.2465, 0.4061) | 0.9084 (0.8765, 0.9368) | 0.4110 (0.3745, 0.4496) | n/a |
| proposed_contextual | 0.0643 (0.0255, 0.1143) | 0.1407 (0.0881, 0.2005) | 0.8463 (0.7911, 0.9016) | 0.2424 (0.2012, 0.2924) | n/a |
| candidate_no_context | 0.0238 (0.0124, 0.0400) | 0.1478 (0.0756, 0.2367) | 0.8110 (0.7703, 0.8546) | 0.2205 (0.1875, 0.2611) | n/a |
| baseline_no_context | 0.0235 (0.0100, 0.0463) | 0.1365 (0.0926, 0.1859) | 0.8872 (0.8530, 0.9183) | 0.2298 (0.2108, 0.2510) | n/a |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0404 | 1.6971 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0071 | -0.0481 |
| proposed_vs_candidate_no_context | naturalness | 0.0353 | 0.0435 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0511 | 3.3750 |
| proposed_vs_candidate_no_context | context_overlap | 0.0155 | 0.3507 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0278 | -0.4000 |
| proposed_vs_candidate_no_context | persona_style | 0.0755 | 0.1638 |
| proposed_vs_candidate_no_context | distinct1 | 0.0285 | 0.0307 |
| proposed_vs_candidate_no_context | length_score | 0.1194 | 0.3644 |
| proposed_vs_candidate_no_context | sentence_score | 0.0000 | 0.0000 |
| proposed_vs_candidate_no_context | overall_quality | 0.0219 | 0.0994 |
| proposed_vs_baseline_no_context | context_relevance | 0.0407 | 1.7305 |
| proposed_vs_baseline_no_context | persona_consistency | 0.0041 | 0.0303 |
| proposed_vs_baseline_no_context | naturalness | -0.0409 | -0.0461 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0524 | 3.7727 |
| proposed_vs_baseline_no_context | context_overlap | 0.0135 | 0.2927 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 0.0111 | 0.3636 |
| proposed_vs_baseline_no_context | persona_style | -0.0238 | -0.0424 |
| proposed_vs_baseline_no_context | distinct1 | -0.0245 | -0.0249 |
| proposed_vs_baseline_no_context | length_score | -0.0972 | -0.1786 |
| proposed_vs_baseline_no_context | sentence_score | -0.1167 | -0.1366 |
| proposed_vs_baseline_no_context | overall_quality | 0.0126 | 0.0547 |
| controlled_vs_proposed_raw | context_relevance | 0.2493 | 3.8799 |
| controlled_vs_proposed_raw | persona_consistency | 0.1375 | 0.9776 |
| controlled_vs_proposed_raw | naturalness | 0.0486 | 0.0574 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.3295 | 4.9714 |
| controlled_vs_proposed_raw | context_overlap | 0.0621 | 1.0427 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1647 | 3.9524 |
| controlled_vs_proposed_raw | persona_style | 0.0289 | 0.0539 |
| controlled_vs_proposed_raw | distinct1 | -0.0005 | -0.0005 |
| controlled_vs_proposed_raw | length_score | 0.1889 | 0.4224 |
| controlled_vs_proposed_raw | sentence_score | 0.1167 | 0.1582 |
| controlled_vs_proposed_raw | overall_quality | 0.1770 | 0.7301 |
| controlled_vs_candidate_no_context | context_relevance | 0.2897 | 12.1616 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1304 | 0.8824 |
| controlled_vs_candidate_no_context | naturalness | 0.0839 | 0.1035 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3807 | 25.1250 |
| controlled_vs_candidate_no_context | context_overlap | 0.0775 | 1.7591 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1369 | 1.9714 |
| controlled_vs_candidate_no_context | persona_style | 0.1044 | 0.2265 |
| controlled_vs_candidate_no_context | distinct1 | 0.0280 | 0.0302 |
| controlled_vs_candidate_no_context | length_score | 0.3083 | 0.9407 |
| controlled_vs_candidate_no_context | sentence_score | 0.1167 | 0.1582 |
| controlled_vs_candidate_no_context | overall_quality | 0.1989 | 0.9020 |
| controlled_vs_baseline_no_context | context_relevance | 0.2900 | 12.3246 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1417 | 1.0375 |
| controlled_vs_baseline_no_context | naturalness | 0.0077 | 0.0087 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3819 | 27.5000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0755 | 1.6407 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1758 | 5.7532 |
| controlled_vs_baseline_no_context | persona_style | 0.0051 | 0.0092 |
| controlled_vs_baseline_no_context | distinct1 | -0.0250 | -0.0254 |
| controlled_vs_baseline_no_context | length_score | 0.0917 | 0.1684 |
| controlled_vs_baseline_no_context | sentence_score | 0.0000 | 0.0000 |
| controlled_vs_baseline_no_context | overall_quality | 0.1895 | 0.8248 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0584 | -0.1862 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0457 | 0.1644 |
| controlled_alt_vs_controlled_default | naturalness | 0.0135 | 0.0151 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0751 | -0.1898 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0194 | -0.1592 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0377 | 0.1827 |
| controlled_alt_vs_controlled_default | persona_style | 0.0779 | 0.1377 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0103 | -0.0107 |
| controlled_alt_vs_controlled_default | length_score | 0.0556 | 0.0873 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0583 | 0.0683 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0084 | -0.0199 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1909 | 2.9711 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1833 | 1.3027 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0621 | 0.0733 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2544 | 3.8381 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0427 | 0.7175 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2024 | 4.8571 |
| controlled_alt_vs_proposed_raw | persona_style | 0.1068 | 0.1990 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0108 | -0.0113 |
| controlled_alt_vs_proposed_raw | length_score | 0.2444 | 0.5466 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1750 | 0.2373 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1686 | 0.6956 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2313 | 9.7105 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1761 | 1.1919 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0974 | 0.1201 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3056 | 20.1667 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0581 | 1.3199 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1746 | 2.5143 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.1823 | 0.3954 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0178 | 0.0191 |
| controlled_alt_vs_candidate_no_context | length_score | 0.3639 | 1.1102 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1750 | 0.2373 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1905 | 0.8641 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.2316 | 9.8431 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.1874 | 1.3724 |
| controlled_alt_vs_baseline_no_context | naturalness | 0.0212 | 0.0239 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.3068 | 22.0909 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0562 | 1.2203 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.2135 | 6.9870 |
| controlled_alt_vs_baseline_no_context | persona_style | 0.0830 | 0.1481 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0352 | -0.0359 |
| controlled_alt_vs_baseline_no_context | length_score | 0.1472 | 0.2704 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.0583 | 0.0683 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.1812 | 0.7884 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2900 | 12.3246 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1417 | 1.0375 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0.0077 | 0.0087 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3819 | 27.5000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0755 | 1.6407 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1758 | 5.7532 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0051 | 0.0092 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0250 | -0.0254 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.0917 | 0.1684 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0000 | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1895 | 0.8248 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0404 | (-0.0015, 0.0923) | 0.0343 | 0.0404 | (-0.0094, 0.0976) | 0.0993 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0071 | (-0.0865, 0.0667) | 0.5443 | -0.0071 | (-0.0615, 0.0365) | 0.5680 |
| proposed_vs_candidate_no_context | naturalness | 0.0353 | (0.0036, 0.0738) | 0.0113 | 0.0353 | (0.0001, 0.0754) | 0.0237 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0511 | (-0.0019, 0.1256) | 0.0513 | 0.0511 | (-0.0152, 0.1276) | 0.0970 |
| proposed_vs_candidate_no_context | context_overlap | 0.0155 | (-0.0016, 0.0355) | 0.0407 | 0.0155 | (0.0030, 0.0293) | 0.0223 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0278 | (-0.1111, 0.0556) | 0.8240 | -0.0278 | (-0.0833, 0.0000) | 1.0000 |
| proposed_vs_candidate_no_context | persona_style | 0.0755 | (0.0000, 0.1988) | 0.1147 | 0.0755 | (0.0000, 0.1852) | 0.0907 |
| proposed_vs_candidate_no_context | distinct1 | 0.0285 | (0.0091, 0.0508) | 0.0013 | 0.0285 | (0.0077, 0.0487) | 0.0037 |
| proposed_vs_candidate_no_context | length_score | 0.1194 | (-0.0083, 0.2500) | 0.0327 | 0.1194 | (-0.0139, 0.2821) | 0.0663 |
| proposed_vs_candidate_no_context | sentence_score | 0.0000 | (-0.1167, 0.1167) | 0.6047 | 0.0000 | (-0.0808, 0.0808) | 0.6420 |
| proposed_vs_candidate_no_context | overall_quality | 0.0219 | (-0.0201, 0.0762) | 0.1950 | 0.0219 | (-0.0236, 0.0653) | 0.1680 |
| proposed_vs_baseline_no_context | context_relevance | 0.0407 | (-0.0018, 0.0930) | 0.0307 | 0.0407 | (0.0037, 0.0929) | 0.0113 |
| proposed_vs_baseline_no_context | persona_consistency | 0.0041 | (-0.0648, 0.0783) | 0.4537 | 0.0041 | (-0.0803, 0.0857) | 0.4803 |
| proposed_vs_baseline_no_context | naturalness | -0.0409 | (-0.1101, 0.0281) | 0.8570 | -0.0409 | (-0.1354, 0.0460) | 0.7977 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0524 | (0.0000, 0.1212) | 0.0300 | 0.0524 | (0.0069, 0.1212) | 0.0167 |
| proposed_vs_baseline_no_context | context_overlap | 0.0135 | (-0.0070, 0.0327) | 0.0960 | 0.0135 | (-0.0057, 0.0323) | 0.0837 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 0.0111 | (-0.0611, 0.0917) | 0.4427 | 0.0111 | (-0.0700, 0.1111) | 0.4700 |
| proposed_vs_baseline_no_context | persona_style | -0.0238 | (-0.1641, 0.0896) | 0.5947 | -0.0238 | (-0.1969, 0.1150) | 0.6377 |
| proposed_vs_baseline_no_context | distinct1 | -0.0245 | (-0.0558, 0.0068) | 0.9390 | -0.0245 | (-0.0599, 0.0084) | 0.9283 |
| proposed_vs_baseline_no_context | length_score | -0.0972 | (-0.3695, 0.1750) | 0.7667 | -0.0972 | (-0.4611, 0.2257) | 0.7133 |
| proposed_vs_baseline_no_context | sentence_score | -0.1167 | (-0.2625, 0.0292) | 0.9613 | -0.1167 | (-0.2917, 0.0955) | 0.8927 |
| proposed_vs_baseline_no_context | overall_quality | 0.0126 | (-0.0385, 0.0691) | 0.3397 | 0.0126 | (-0.0490, 0.0777) | 0.3780 |
| controlled_vs_proposed_raw | context_relevance | 0.2493 | (0.1866, 0.3273) | 0.0000 | 0.2493 | (0.2019, 0.3383) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1375 | (0.0652, 0.2023) | 0.0000 | 0.1375 | (0.0492, 0.2127) | 0.0020 |
| controlled_vs_proposed_raw | naturalness | 0.0486 | (-0.0158, 0.1146) | 0.0727 | 0.0486 | (-0.0291, 0.1256) | 0.0963 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.3295 | (0.2449, 0.4432) | 0.0000 | 0.3295 | (0.2692, 0.4500) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0621 | (0.0383, 0.0855) | 0.0000 | 0.0621 | (0.0380, 0.0874) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1647 | (0.0833, 0.2369) | 0.0000 | 0.1647 | (0.0595, 0.2452) | 0.0017 |
| controlled_vs_proposed_raw | persona_style | 0.0289 | (-0.0837, 0.1684) | 0.3577 | 0.0289 | (-0.0849, 0.1823) | 0.3107 |
| controlled_vs_proposed_raw | distinct1 | -0.0005 | (-0.0294, 0.0276) | 0.5143 | -0.0005 | (-0.0370, 0.0397) | 0.5377 |
| controlled_vs_proposed_raw | length_score | 0.1889 | (-0.0500, 0.4250) | 0.0620 | 0.1889 | (-0.0933, 0.4667) | 0.0910 |
| controlled_vs_proposed_raw | sentence_score | 0.1167 | (0.0000, 0.2333) | 0.0533 | 0.1167 | (0.0250, 0.2423) | 0.0247 |
| controlled_vs_proposed_raw | overall_quality | 0.1770 | (0.1260, 0.2229) | 0.0000 | 0.1770 | (0.1277, 0.2274) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2897 | (0.2298, 0.3631) | 0.0000 | 0.2897 | (0.2249, 0.3835) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1304 | (0.0337, 0.2157) | 0.0090 | 0.1304 | (0.0407, 0.2067) | 0.0020 |
| controlled_vs_candidate_no_context | naturalness | 0.0839 | (0.0300, 0.1381) | 0.0013 | 0.0839 | (0.0160, 0.1463) | 0.0077 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3807 | (0.2973, 0.4817) | 0.0000 | 0.3807 | (0.2937, 0.5083) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0775 | (0.0597, 0.0933) | 0.0000 | 0.0775 | (0.0592, 0.0949) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1369 | (0.0222, 0.2321) | 0.0107 | 0.1369 | (0.0366, 0.2242) | 0.0043 |
| controlled_vs_candidate_no_context | persona_style | 0.1044 | (-0.0562, 0.2886) | 0.1120 | 0.1044 | (-0.0290, 0.2566) | 0.0663 |
| controlled_vs_candidate_no_context | distinct1 | 0.0280 | (-0.0052, 0.0580) | 0.0467 | 0.0280 | (-0.0133, 0.0633) | 0.1003 |
| controlled_vs_candidate_no_context | length_score | 0.3083 | (0.1083, 0.5083) | 0.0010 | 0.3083 | (0.0633, 0.5296) | 0.0060 |
| controlled_vs_candidate_no_context | sentence_score | 0.1167 | (0.0000, 0.2333) | 0.0527 | 0.1167 | (-0.0292, 0.2545) | 0.0817 |
| controlled_vs_candidate_no_context | overall_quality | 0.1989 | (0.1511, 0.2387) | 0.0000 | 0.1989 | (0.1674, 0.2339) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2900 | (0.2281, 0.3731) | 0.0000 | 0.2900 | (0.2282, 0.3771) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1417 | (0.1037, 0.1773) | 0.0000 | 0.1417 | (0.0973, 0.1711) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | 0.0077 | (-0.0298, 0.0524) | 0.3867 | 0.0077 | (-0.0351, 0.0518) | 0.3723 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3819 | (0.2986, 0.4799) | 0.0000 | 0.3819 | (0.2986, 0.4960) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0755 | (0.0610, 0.0934) | 0.0000 | 0.0755 | (0.0626, 0.0972) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1758 | (0.1333, 0.2167) | 0.0000 | 0.1758 | (0.1276, 0.2099) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | 0.0051 | (-0.0794, 0.0962) | 0.4643 | 0.0051 | (-0.0852, 0.0802) | 0.4633 |
| controlled_vs_baseline_no_context | distinct1 | -0.0250 | (-0.0532, 0.0055) | 0.9490 | -0.0250 | (-0.0609, 0.0112) | 0.9123 |
| controlled_vs_baseline_no_context | length_score | 0.0917 | (-0.0917, 0.2944) | 0.1767 | 0.0917 | (-0.1282, 0.3191) | 0.2383 |
| controlled_vs_baseline_no_context | sentence_score | 0.0000 | (-0.1458, 0.1458) | 0.5810 | 0.0000 | (-0.1167, 0.1458) | 0.6223 |
| controlled_vs_baseline_no_context | overall_quality | 0.1895 | (0.1561, 0.2246) | 0.0000 | 0.1895 | (0.1545, 0.2285) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | -0.0584 | (-0.1433, 0.0065) | 0.9490 | -0.0584 | (-0.1517, 0.0048) | 0.9627 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0457 | (-0.0383, 0.1428) | 0.1593 | 0.0457 | (-0.0496, 0.1553) | 0.1790 |
| controlled_alt_vs_controlled_default | naturalness | 0.0135 | (-0.0337, 0.0622) | 0.2927 | 0.0135 | (-0.0297, 0.0652) | 0.2757 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | -0.0751 | (-0.1894, 0.0158) | 0.9287 | -0.0751 | (-0.2000, 0.0146) | 0.9363 |
| controlled_alt_vs_controlled_default | context_overlap | -0.0194 | (-0.0399, 0.0037) | 0.9457 | -0.0194 | (-0.0407, 0.0026) | 0.9587 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0377 | (-0.0528, 0.1421) | 0.2373 | 0.0377 | (-0.0623, 0.1451) | 0.2427 |
| controlled_alt_vs_controlled_default | persona_style | 0.0779 | (-0.0078, 0.1846) | 0.0420 | 0.0779 | (0.0085, 0.1682) | 0.0250 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0103 | (-0.0295, 0.0072) | 0.8533 | -0.0103 | (-0.0323, 0.0088) | 0.8323 |
| controlled_alt_vs_controlled_default | length_score | 0.0556 | (-0.1500, 0.2722) | 0.3080 | 0.0556 | (-0.1513, 0.2967) | 0.3137 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0583 | (0.0000, 0.1458) | 0.1000 | 0.0583 | (0.0000, 0.1458) | 0.0977 |
| controlled_alt_vs_controlled_default | overall_quality | -0.0084 | (-0.0464, 0.0335) | 0.6693 | -0.0084 | (-0.0479, 0.0357) | 0.6520 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1909 | (0.1174, 0.2515) | 0.0000 | 0.1909 | (0.1370, 0.2429) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1833 | (0.0820, 0.2973) | 0.0003 | 0.1833 | (0.0702, 0.3064) | 0.0007 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0621 | (0.0094, 0.1131) | 0.0110 | 0.0621 | (0.0016, 0.1298) | 0.0227 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2544 | (0.1546, 0.3390) | 0.0000 | 0.2544 | (0.1830, 0.3258) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0427 | (0.0222, 0.0631) | 0.0000 | 0.0427 | (0.0262, 0.0618) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.2024 | (0.0940, 0.3131) | 0.0003 | 0.2024 | (0.0769, 0.3324) | 0.0003 |
| controlled_alt_vs_proposed_raw | persona_style | 0.1068 | (-0.0069, 0.2726) | 0.1187 | 0.1068 | (-0.0064, 0.2726) | 0.0827 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0108 | (-0.0354, 0.0117) | 0.8110 | -0.0108 | (-0.0382, 0.0149) | 0.7770 |
| controlled_alt_vs_proposed_raw | length_score | 0.2444 | (0.0528, 0.4306) | 0.0077 | 0.2444 | (0.0166, 0.4788) | 0.0180 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1750 | (0.0292, 0.2917) | 0.0107 | 0.1750 | (0.0750, 0.2864) | 0.0003 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1686 | (0.0973, 0.2350) | 0.0000 | 0.1686 | (0.0967, 0.2391) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2313 | (0.1949, 0.2691) | 0.0000 | 0.2313 | (0.2031, 0.2585) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.1761 | (0.0421, 0.3001) | 0.0043 | 0.1761 | (0.0552, 0.3130) | 0.0007 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0974 | (0.0529, 0.1402) | 0.0000 | 0.0974 | (0.0463, 0.1547) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3056 | (0.2551, 0.3586) | 0.0000 | 0.3056 | (0.2672, 0.3409) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0581 | (0.0352, 0.0826) | 0.0000 | 0.0581 | (0.0409, 0.0760) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1746 | (0.0278, 0.3075) | 0.0093 | 0.1746 | (0.0619, 0.3181) | 0.0017 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.1823 | (0.0278, 0.3543) | 0.0073 | 0.1823 | (0.0505, 0.3534) | 0.0173 |
| controlled_alt_vs_candidate_no_context | distinct1 | 0.0178 | (-0.0093, 0.0457) | 0.1030 | 0.0178 | (-0.0100, 0.0465) | 0.1027 |
| controlled_alt_vs_candidate_no_context | length_score | 0.3639 | (0.1944, 0.5361) | 0.0000 | 0.3639 | (0.1667, 0.5667) | 0.0000 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1750 | (0.0875, 0.2625) | 0.0007 | 0.1750 | (0.0808, 0.2864) | 0.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1905 | (0.1422, 0.2449) | 0.0000 | 0.1905 | (0.1416, 0.2503) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_relevance | 0.2316 | (0.1870, 0.2755) | 0.0000 | 0.2316 | (0.2013, 0.2610) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 0.1874 | (0.1113, 0.2811) | 0.0000 | 0.1874 | (0.0987, 0.2856) | 0.0000 |
| controlled_alt_vs_baseline_no_context | naturalness | 0.0212 | (-0.0301, 0.0725) | 0.2123 | 0.0212 | (-0.0408, 0.0766) | 0.2403 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 0.3068 | (0.2525, 0.3636) | 0.0000 | 0.3068 | (0.2645, 0.3474) | 0.0000 |
| controlled_alt_vs_baseline_no_context | context_overlap | 0.0562 | (0.0388, 0.0746) | 0.0000 | 0.0562 | (0.0419, 0.0713) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 0.2135 | (0.1306, 0.3111) | 0.0000 | 0.2135 | (0.1154, 0.3201) | 0.0000 |
| controlled_alt_vs_baseline_no_context | persona_style | 0.0830 | (0.0000, 0.1821) | 0.0360 | 0.0830 | (0.0000, 0.1783) | 0.0913 |
| controlled_alt_vs_baseline_no_context | distinct1 | -0.0352 | (-0.0643, -0.0024) | 0.9777 | -0.0352 | (-0.0641, 0.0016) | 0.9740 |
| controlled_alt_vs_baseline_no_context | length_score | 0.1472 | (-0.0668, 0.3722) | 0.0947 | 0.1472 | (-0.1154, 0.4091) | 0.1383 |
| controlled_alt_vs_baseline_no_context | sentence_score | 0.0583 | (-0.0875, 0.1757) | 0.2600 | 0.0583 | (-0.0583, 0.2042) | 0.2767 |
| controlled_alt_vs_baseline_no_context | overall_quality | 0.1812 | (0.1377, 0.2266) | 0.0000 | 0.1812 | (0.1365, 0.2208) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2900 | (0.2294, 0.3669) | 0.0000 | 0.2900 | (0.2282, 0.3745) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1417 | (0.1041, 0.1768) | 0.0000 | 0.1417 | (0.0958, 0.1707) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0.0077 | (-0.0305, 0.0544) | 0.3807 | 0.0077 | (-0.0360, 0.0529) | 0.3770 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3819 | (0.2992, 0.4823) | 0.0000 | 0.3819 | (0.2955, 0.4909) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0755 | (0.0615, 0.0937) | 0.0000 | 0.0755 | (0.0628, 0.0963) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1758 | (0.1341, 0.2139) | 0.0000 | 0.1758 | (0.1276, 0.2103) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0051 | (-0.0781, 0.0962) | 0.4620 | 0.0051 | (-0.0852, 0.0863) | 0.4460 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0250 | (-0.0517, 0.0057) | 0.9483 | -0.0250 | (-0.0592, 0.0126) | 0.9127 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.0917 | (-0.0806, 0.2917) | 0.1783 | 0.0917 | (-0.1303, 0.3278) | 0.2403 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0000 | (-0.1458, 0.1458) | 0.5943 | 0.0000 | (-0.1167, 0.1500) | 0.6070 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1895 | (0.1553, 0.2259) | 0.0000 | 0.1895 | (0.1542, 0.2267) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 5 | 2 | 5 | 0.6250 | 0.7143 |
| proposed_vs_candidate_no_context | persona_consistency | 3 | 2 | 7 | 0.5417 | 0.6000 |
| proposed_vs_candidate_no_context | naturalness | 6 | 1 | 5 | 0.7083 | 0.8571 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 4 | 1 | 7 | 0.6250 | 0.8000 |
| proposed_vs_candidate_no_context | context_overlap | 5 | 2 | 5 | 0.6250 | 0.7143 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 1 | 2 | 9 | 0.4583 | 0.3333 |
| proposed_vs_candidate_no_context | persona_style | 2 | 0 | 10 | 0.5833 | 1.0000 |
| proposed_vs_candidate_no_context | distinct1 | 5 | 0 | 7 | 0.7083 | 1.0000 |
| proposed_vs_candidate_no_context | length_score | 6 | 1 | 5 | 0.7083 | 0.8571 |
| proposed_vs_candidate_no_context | sentence_score | 2 | 2 | 8 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | overall_quality | 4 | 3 | 5 | 0.5417 | 0.5714 |
| proposed_vs_baseline_no_context | context_relevance | 7 | 5 | 0 | 0.5833 | 0.5833 |
| proposed_vs_baseline_no_context | persona_consistency | 4 | 3 | 5 | 0.5417 | 0.5714 |
| proposed_vs_baseline_no_context | naturalness | 6 | 6 | 0 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 4 | 1 | 7 | 0.6250 | 0.8000 |
| proposed_vs_baseline_no_context | context_overlap | 6 | 6 | 0 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 2 | 2 | 8 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | persona_style | 2 | 2 | 8 | 0.5000 | 0.5000 |
| proposed_vs_baseline_no_context | distinct1 | 3 | 6 | 3 | 0.3750 | 0.3333 |
| proposed_vs_baseline_no_context | length_score | 5 | 7 | 0 | 0.4167 | 0.4167 |
| proposed_vs_baseline_no_context | sentence_score | 2 | 6 | 4 | 0.3333 | 0.2500 |
| proposed_vs_baseline_no_context | overall_quality | 6 | 6 | 0 | 0.5000 | 0.5000 |
| controlled_vs_proposed_raw | context_relevance | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_proposed_raw | persona_consistency | 10 | 1 | 1 | 0.8750 | 0.9091 |
| controlled_vs_proposed_raw | naturalness | 7 | 5 | 0 | 0.5833 | 0.5833 |
| controlled_vs_proposed_raw | context_keyword_coverage | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_proposed_raw | context_overlap | 10 | 2 | 0 | 0.8333 | 0.8333 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 10 | 1 | 1 | 0.8750 | 0.9091 |
| controlled_vs_proposed_raw | persona_style | 3 | 2 | 7 | 0.5417 | 0.6000 |
| controlled_vs_proposed_raw | distinct1 | 6 | 5 | 1 | 0.5417 | 0.5455 |
| controlled_vs_proposed_raw | length_score | 7 | 5 | 0 | 0.5833 | 0.5833 |
| controlled_vs_proposed_raw | sentence_score | 5 | 1 | 6 | 0.6667 | 0.8333 |
| controlled_vs_proposed_raw | overall_quality | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_relevance | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 10 | 2 | 0 | 0.8333 | 0.8333 |
| controlled_vs_candidate_no_context | naturalness | 8 | 4 | 0 | 0.6667 | 0.6667 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 10 | 2 | 0 | 0.8333 | 0.8333 |
| controlled_vs_candidate_no_context | persona_style | 4 | 2 | 6 | 0.5833 | 0.6667 |
| controlled_vs_candidate_no_context | distinct1 | 9 | 3 | 0 | 0.7500 | 0.7500 |
| controlled_vs_candidate_no_context | length_score | 8 | 3 | 1 | 0.7083 | 0.7273 |
| controlled_vs_candidate_no_context | sentence_score | 5 | 1 | 6 | 0.6667 | 0.8333 |
| controlled_vs_candidate_no_context | overall_quality | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_relevance | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 11 | 0 | 1 | 0.9583 | 1.0000 |
| controlled_vs_baseline_no_context | naturalness | 5 | 7 | 0 | 0.4167 | 0.4167 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 11 | 0 | 1 | 0.9583 | 1.0000 |
| controlled_vs_baseline_no_context | persona_style | 1 | 2 | 9 | 0.4583 | 0.3333 |
| controlled_vs_baseline_no_context | distinct1 | 4 | 7 | 1 | 0.3750 | 0.3636 |
| controlled_vs_baseline_no_context | length_score | 6 | 6 | 0 | 0.5000 | 0.5000 |
| controlled_vs_baseline_no_context | sentence_score | 3 | 3 | 6 | 0.5000 | 0.5000 |
| controlled_vs_baseline_no_context | overall_quality | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 5 | 7 | 0 | 0.4167 | 0.4167 |
| controlled_alt_vs_controlled_default | persona_consistency | 4 | 3 | 5 | 0.5417 | 0.5714 |
| controlled_alt_vs_controlled_default | naturalness | 8 | 4 | 0 | 0.6667 | 0.6667 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 2 | 4 | 6 | 0.4167 | 0.3333 |
| controlled_alt_vs_controlled_default | context_overlap | 4 | 8 | 0 | 0.3333 | 0.3333 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 3 | 2 | 7 | 0.5417 | 0.6000 |
| controlled_alt_vs_controlled_default | persona_style | 3 | 1 | 8 | 0.5833 | 0.7500 |
| controlled_alt_vs_controlled_default | distinct1 | 5 | 6 | 1 | 0.4583 | 0.4545 |
| controlled_alt_vs_controlled_default | length_score | 7 | 4 | 1 | 0.6250 | 0.6364 |
| controlled_alt_vs_controlled_default | sentence_score | 2 | 0 | 10 | 0.5833 | 1.0000 |
| controlled_alt_vs_controlled_default | overall_quality | 4 | 8 | 0 | 0.3333 | 0.3333 |
| controlled_alt_vs_proposed_raw | context_relevance | 11 | 1 | 0 | 0.9167 | 0.9167 |
| controlled_alt_vs_proposed_raw | persona_consistency | 9 | 1 | 2 | 0.8333 | 0.9000 |
| controlled_alt_vs_proposed_raw | naturalness | 9 | 3 | 0 | 0.7500 | 0.7500 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 11 | 1 | 0 | 0.9167 | 0.9167 |
| controlled_alt_vs_proposed_raw | context_overlap | 11 | 1 | 0 | 0.9167 | 0.9167 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 9 | 1 | 2 | 0.8333 | 0.9000 |
| controlled_alt_vs_proposed_raw | persona_style | 2 | 1 | 9 | 0.5417 | 0.6667 |
| controlled_alt_vs_proposed_raw | distinct1 | 3 | 6 | 3 | 0.3750 | 0.3333 |
| controlled_alt_vs_proposed_raw | length_score | 10 | 2 | 0 | 0.8333 | 0.8333 |
| controlled_alt_vs_proposed_raw | sentence_score | 7 | 1 | 4 | 0.7500 | 0.8750 |
| controlled_alt_vs_proposed_raw | overall_quality | 11 | 1 | 0 | 0.9167 | 0.9167 |
| controlled_alt_vs_candidate_no_context | context_relevance | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 9 | 2 | 1 | 0.7917 | 0.8182 |
| controlled_alt_vs_candidate_no_context | naturalness | 10 | 2 | 0 | 0.8333 | 0.8333 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 11 | 1 | 0 | 0.9167 | 0.9167 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 9 | 2 | 1 | 0.7917 | 0.8182 |
| controlled_alt_vs_candidate_no_context | persona_style | 4 | 0 | 8 | 0.6667 | 1.0000 |
| controlled_alt_vs_candidate_no_context | distinct1 | 6 | 5 | 1 | 0.5417 | 0.5455 |
| controlled_alt_vs_candidate_no_context | length_score | 10 | 2 | 0 | 0.8333 | 0.8333 |
| controlled_alt_vs_candidate_no_context | sentence_score | 6 | 0 | 6 | 0.7500 | 1.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | context_relevance | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | persona_consistency | 10 | 0 | 2 | 0.9167 | 1.0000 |
| controlled_alt_vs_baseline_no_context | naturalness | 8 | 4 | 0 | 0.6667 | 0.6667 |
| controlled_alt_vs_baseline_no_context | context_keyword_coverage | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | context_overlap | 12 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_baseline_no_context | persona_keyword_coverage | 10 | 0 | 2 | 0.9167 | 1.0000 |
| controlled_alt_vs_baseline_no_context | persona_style | 3 | 0 | 9 | 0.6250 | 1.0000 |
| controlled_alt_vs_baseline_no_context | distinct1 | 3 | 9 | 0 | 0.2500 | 0.2500 |
| controlled_alt_vs_baseline_no_context | length_score | 8 | 4 | 0 | 0.6667 | 0.6667 |
| controlled_alt_vs_baseline_no_context | sentence_score | 4 | 2 | 6 | 0.5833 | 0.6667 |
| controlled_alt_vs_baseline_no_context | overall_quality | 12 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 12 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 11 | 0 | 1 | 0.9583 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 5 | 7 | 0 | 0.4167 | 0.4167 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 12 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 12 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 11 | 0 | 1 | 0.9583 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 1 | 2 | 9 | 0.4583 | 0.3333 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 4 | 7 | 1 | 0.3750 | 0.3636 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 6 | 6 | 0 | 0.5000 | 0.5000 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 3 | 3 | 6 | 0.5000 | 0.5000 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 12 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.1667 | 0.5833 | 0.4167 |
| proposed_contextual_controlled_alt | 0.0000 | 0.0000 | 0.1667 | 0.5833 | 0.4167 |
| proposed_contextual | 0.0000 | 0.0000 | 0.4167 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.5833 | 0.0000 | 0.0000 |
| baseline_no_context | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `7`
- Unique template signatures: `11`
- Template signature ratio: `0.9167`
- Effective sample size by source clustering: `6.55`
- Effective sample size by template-signature clustering: `10.29`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
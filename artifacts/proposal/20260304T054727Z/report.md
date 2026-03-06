# Proposal Alignment Evaluation Report

- Run ID: `20260304T054727Z`
- Generated: `2026-03-04T05:49:35.489222+00:00`
- Scenarios: `artifacts\proposal\20260304T054727Z\scenarios.jsonl`
- Scenario count: `112`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off
- `baseline_no_context`: model `phi3:mini`, dynamic_context=off, response_control=off
- `baseline_no_context_phi3_latest`: model `phi3:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2548 (0.2418, 0.2698) | 0.3656 (0.3393, 0.3947) | 0.9129 (0.9019, 0.9234) | 0.3910 (0.3804, 0.4020) | 0.0871 |
| proposed_contextual | 0.0837 (0.0655, 0.1039) | 0.1555 (0.1294, 0.1841) | 0.7990 (0.7871, 0.8125) | 0.2303 (0.2157, 0.2452) | 0.0695 |
| candidate_no_context | 0.0238 (0.0185, 0.0297) | 0.1633 (0.1373, 0.1910) | 0.8110 (0.7968, 0.8268) | 0.2077 (0.1967, 0.2189) | 0.0432 |
| baseline_no_context | 0.0448 (0.0359, 0.0543) | 0.1924 (0.1706, 0.2153) | 0.8774 (0.8662, 0.8882) | 0.2382 (0.2295, 0.2473) | 0.0535 |
| baseline_no_context_phi3_latest | 0.0499 (0.0396, 0.0604) | 0.1847 (0.1627, 0.2059) | 0.8863 (0.8743, 0.8977) | 0.2395 (0.2309, 0.2492) | 0.0552 |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0599 | 2.5150 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0078 | -0.0477 |
| proposed_vs_candidate_no_context | naturalness | -0.0120 | -0.0148 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0778 | 4.5399 |
| proposed_vs_candidate_no_context | context_overlap | 0.0180 | 0.4578 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0100 | -0.1284 |
| proposed_vs_candidate_no_context | persona_style | 0.0010 | 0.0020 |
| proposed_vs_candidate_no_context | distinct1 | 0.0017 | 0.0018 |
| proposed_vs_candidate_no_context | length_score | -0.0494 | -0.1653 |
| proposed_vs_candidate_no_context | sentence_score | -0.0277 | -0.0357 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0263 | 0.6089 |
| proposed_vs_candidate_no_context | overall_quality | 0.0226 | 0.1088 |
| proposed_vs_baseline_no_context | context_relevance | 0.0388 | 0.8663 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0369 | -0.1917 |
| proposed_vs_baseline_no_context | naturalness | -0.0784 | -0.0893 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0484 | 1.0404 |
| proposed_vs_baseline_no_context | context_overlap | 0.0165 | 0.4037 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0274 | -0.2879 |
| proposed_vs_baseline_no_context | persona_style | -0.0748 | -0.1287 |
| proposed_vs_baseline_no_context | distinct1 | -0.0377 | -0.0388 |
| proposed_vs_baseline_no_context | length_score | -0.2542 | -0.5047 |
| proposed_vs_baseline_no_context | sentence_score | -0.1246 | -0.1429 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0160 | 0.2996 |
| proposed_vs_baseline_no_context | overall_quality | -0.0080 | -0.0335 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.0338 | 0.6778 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | -0.0292 | -0.1579 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0873 | -0.0985 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0417 | 0.7819 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0155 | 0.3688 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | -0.0179 | -0.2088 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | -0.0742 | -0.1278 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0431 | -0.0440 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.2851 | -0.5334 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | -0.1308 | -0.1490 |
| proposed_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0144 | 0.2606 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | -0.0093 | -0.0388 |
| controlled_vs_proposed_raw | context_relevance | 0.1711 | 2.0446 |
| controlled_vs_proposed_raw | persona_consistency | 0.2100 | 1.3505 |
| controlled_vs_proposed_raw | naturalness | 0.1139 | 0.1425 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2263 | 2.3835 |
| controlled_vs_proposed_raw | context_overlap | 0.0422 | 0.7360 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2404 | 3.5448 |
| controlled_vs_proposed_raw | persona_style | 0.0887 | 0.1751 |
| controlled_vs_proposed_raw | distinct1 | 0.0047 | 0.0050 |
| controlled_vs_proposed_raw | length_score | 0.4461 | 1.7888 |
| controlled_vs_proposed_raw | sentence_score | 0.2277 | 0.3047 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0175 | 0.2520 |
| controlled_vs_proposed_raw | overall_quality | 0.1608 | 0.6982 |
| controlled_vs_candidate_no_context | context_relevance | 0.2310 | 9.7019 |
| controlled_vs_candidate_no_context | persona_consistency | 0.2023 | 1.2384 |
| controlled_vs_candidate_no_context | naturalness | 0.1019 | 0.1256 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3041 | 17.7443 |
| controlled_vs_candidate_no_context | context_overlap | 0.0603 | 1.5307 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2304 | 2.9612 |
| controlled_vs_candidate_no_context | persona_style | 0.0897 | 0.1774 |
| controlled_vs_candidate_no_context | distinct1 | 0.0064 | 0.0068 |
| controlled_vs_candidate_no_context | length_score | 0.3967 | 1.3277 |
| controlled_vs_candidate_no_context | sentence_score | 0.2000 | 0.2581 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0438 | 1.0144 |
| controlled_vs_candidate_no_context | overall_quality | 0.1834 | 0.8829 |
| controlled_vs_baseline_no_context | context_relevance | 0.2099 | 4.6820 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1731 | 0.8998 |
| controlled_vs_baseline_no_context | naturalness | 0.0355 | 0.0404 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.2747 | 5.9038 |
| controlled_vs_baseline_no_context | context_overlap | 0.0587 | 1.4368 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2130 | 2.2362 |
| controlled_vs_baseline_no_context | persona_style | 0.0139 | 0.0239 |
| controlled_vs_baseline_no_context | distinct1 | -0.0331 | -0.0340 |
| controlled_vs_baseline_no_context | length_score | 0.1920 | 0.3812 |
| controlled_vs_baseline_no_context | sentence_score | 0.1031 | 0.1183 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0336 | 0.6271 |
| controlled_vs_baseline_no_context | overall_quality | 0.1528 | 0.6412 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2049 | 4.1081 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1809 | 0.9794 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | 0.0265 | 0.0299 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.2680 | 5.0292 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0577 | 1.3763 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.2225 | 2.5957 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0144 | 0.0249 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0384 | -0.0392 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 0.1610 | 0.3012 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0969 | 0.1103 |
| controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0319 | 0.5784 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1515 | 0.6324 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2099 | 4.6820 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1731 | 0.8998 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0.0355 | 0.0404 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.2747 | 5.9038 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0587 | 1.4368 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2130 | 2.2362 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0139 | 0.0239 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0331 | -0.0340 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.1920 | 0.3812 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.1031 | 0.1183 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0336 | 0.6271 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1528 | 0.6412 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2049 | 4.1081 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1809 | 0.9794 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | 0.0265 | 0.0299 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.2680 | 5.0292 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0577 | 1.3763 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.2225 | 2.5957 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0144 | 0.0249 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0384 | -0.0392 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 0.1610 | 0.3012 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0969 | 0.1103 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0319 | 0.5784 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1515 | 0.6324 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0599 | (0.0413, 0.0791) | 0.0000 | 0.0599 | (0.0226, 0.1024) | 0.0000 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0078 | (-0.0361, 0.0198) | 0.7050 | -0.0078 | (-0.0473, 0.0207) | 0.6477 |
| proposed_vs_candidate_no_context | naturalness | -0.0120 | (-0.0299, 0.0048) | 0.9180 | -0.0120 | (-0.0241, 0.0006) | 0.9673 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0778 | (0.0532, 0.1034) | 0.0000 | 0.0778 | (0.0291, 0.1330) | 0.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0180 | (0.0106, 0.0264) | 0.0000 | 0.0180 | (0.0062, 0.0318) | 0.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0100 | (-0.0466, 0.0245) | 0.7130 | -0.0100 | (-0.0559, 0.0238) | 0.6443 |
| proposed_vs_candidate_no_context | persona_style | 0.0010 | (-0.0292, 0.0316) | 0.4460 | 0.0010 | (-0.0394, 0.0409) | 0.3797 |
| proposed_vs_candidate_no_context | distinct1 | 0.0017 | (-0.0063, 0.0102) | 0.3440 | 0.0017 | (-0.0065, 0.0109) | 0.3740 |
| proposed_vs_candidate_no_context | length_score | -0.0494 | (-0.1158, 0.0164) | 0.9280 | -0.0494 | (-0.0961, -0.0080) | 0.9917 |
| proposed_vs_candidate_no_context | sentence_score | -0.0277 | (-0.0679, 0.0098) | 0.9183 | -0.0277 | (-0.0719, 0.0161) | 0.8850 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0263 | (0.0163, 0.0372) | 0.0000 | 0.0263 | (0.0123, 0.0386) | 0.0000 |
| proposed_vs_candidate_no_context | overall_quality | 0.0226 | (0.0068, 0.0394) | 0.0023 | 0.0226 | (-0.0016, 0.0490) | 0.0373 |
| proposed_vs_baseline_no_context | context_relevance | 0.0388 | (0.0190, 0.0597) | 0.0000 | 0.0388 | (-0.0029, 0.0837) | 0.0313 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0369 | (-0.0650, -0.0079) | 0.9950 | -0.0369 | (-0.0841, 0.0086) | 0.9420 |
| proposed_vs_baseline_no_context | naturalness | -0.0784 | (-0.0947, -0.0610) | 1.0000 | -0.0784 | (-0.1065, -0.0518) | 1.0000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0484 | (0.0220, 0.0761) | 0.0000 | 0.0484 | (-0.0030, 0.1061) | 0.0333 |
| proposed_vs_baseline_no_context | context_overlap | 0.0165 | (0.0090, 0.0249) | 0.0000 | 0.0165 | (0.0059, 0.0275) | 0.0007 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | -0.0274 | (-0.0573, 0.0021) | 0.9623 | -0.0274 | (-0.0622, 0.0120) | 0.9137 |
| proposed_vs_baseline_no_context | persona_style | -0.0748 | (-0.1224, -0.0322) | 1.0000 | -0.0748 | (-0.2189, 0.0361) | 0.8747 |
| proposed_vs_baseline_no_context | distinct1 | -0.0377 | (-0.0467, -0.0288) | 1.0000 | -0.0377 | (-0.0558, -0.0161) | 0.9993 |
| proposed_vs_baseline_no_context | length_score | -0.2542 | (-0.3212, -0.1893) | 1.0000 | -0.2542 | (-0.3464, -0.1533) | 1.0000 |
| proposed_vs_baseline_no_context | sentence_score | -0.1246 | (-0.1657, -0.0781) | 1.0000 | -0.1246 | (-0.2058, -0.0402) | 0.9973 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0160 | (0.0031, 0.0295) | 0.0073 | 0.0160 | (-0.0072, 0.0395) | 0.0917 |
| proposed_vs_baseline_no_context | overall_quality | -0.0080 | (-0.0237, 0.0082) | 0.8487 | -0.0080 | (-0.0391, 0.0277) | 0.6840 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.0338 | (0.0131, 0.0547) | 0.0007 | 0.0338 | (-0.0066, 0.0781) | 0.0533 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | -0.0292 | (-0.0580, -0.0006) | 0.9773 | -0.0292 | (-0.0791, 0.0177) | 0.8777 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0873 | (-0.1041, -0.0709) | 1.0000 | -0.0873 | (-0.1116, -0.0624) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0417 | (0.0132, 0.0691) | 0.0013 | 0.0417 | (-0.0097, 0.1005) | 0.0640 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0155 | (0.0079, 0.0243) | 0.0000 | 0.0155 | (0.0023, 0.0299) | 0.0077 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | -0.0179 | (-0.0511, 0.0162) | 0.8357 | -0.0179 | (-0.0548, 0.0274) | 0.8043 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | -0.0742 | (-0.1154, -0.0337) | 1.0000 | -0.0742 | (-0.2162, 0.0276) | 0.8650 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0431 | (-0.0508, -0.0351) | 1.0000 | -0.0431 | (-0.0557, -0.0281) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.2851 | (-0.3491, -0.2220) | 1.0000 | -0.2851 | (-0.3673, -0.1926) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | -0.1308 | (-0.1714, -0.0902) | 1.0000 | -0.1308 | (-0.2031, -0.0594) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0144 | (0.0023, 0.0258) | 0.0063 | 0.0144 | (-0.0055, 0.0367) | 0.0850 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | -0.0093 | (-0.0253, 0.0071) | 0.8647 | -0.0093 | (-0.0400, 0.0246) | 0.7143 |
| controlled_vs_proposed_raw | context_relevance | 0.1711 | (0.1465, 0.1953) | 0.0000 | 0.1711 | (0.1261, 0.2123) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.2100 | (0.1787, 0.2413) | 0.0000 | 0.2100 | (0.1600, 0.2597) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.1139 | (0.0967, 0.1300) | 0.0000 | 0.1139 | (0.0881, 0.1390) | 0.0000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2263 | (0.1947, 0.2567) | 0.0000 | 0.2263 | (0.1663, 0.2748) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0422 | (0.0320, 0.0521) | 0.0000 | 0.0422 | (0.0265, 0.0599) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.2404 | (0.2032, 0.2777) | 0.0000 | 0.2404 | (0.1800, 0.3052) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0887 | (0.0517, 0.1283) | 0.0000 | 0.0887 | (0.0018, 0.2205) | 0.0220 |
| controlled_vs_proposed_raw | distinct1 | 0.0047 | (-0.0028, 0.0123) | 0.1183 | 0.0047 | (-0.0112, 0.0148) | 0.2547 |
| controlled_vs_proposed_raw | length_score | 0.4461 | (0.3780, 0.5092) | 0.0000 | 0.4461 | (0.3500, 0.5432) | 0.0000 |
| controlled_vs_proposed_raw | sentence_score | 0.2277 | (0.1933, 0.2616) | 0.0000 | 0.2277 | (0.1705, 0.2812) | 0.0000 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0175 | (0.0013, 0.0326) | 0.0157 | 0.0175 | (-0.0187, 0.0469) | 0.1433 |
| controlled_vs_proposed_raw | overall_quality | 0.1608 | (0.1447, 0.1773) | 0.0000 | 0.1608 | (0.1282, 0.1905) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2310 | (0.2174, 0.2456) | 0.0000 | 0.2310 | (0.2099, 0.2496) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.2023 | (0.1681, 0.2346) | 0.0000 | 0.2023 | (0.1347, 0.2615) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.1019 | (0.0856, 0.1187) | 0.0000 | 0.1019 | (0.0732, 0.1298) | 0.0000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3041 | (0.2865, 0.3236) | 0.0000 | 0.3041 | (0.2769, 0.3277) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0603 | (0.0539, 0.0669) | 0.0000 | 0.0603 | (0.0523, 0.0696) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.2304 | (0.1901, 0.2712) | 0.0000 | 0.2304 | (0.1488, 0.3102) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0897 | (0.0531, 0.1285) | 0.0000 | 0.0897 | (0.0124, 0.1922) | 0.0043 |
| controlled_vs_candidate_no_context | distinct1 | 0.0064 | (-0.0013, 0.0139) | 0.0493 | 0.0064 | (-0.0044, 0.0173) | 0.1230 |
| controlled_vs_candidate_no_context | length_score | 0.3967 | (0.3295, 0.4625) | 0.0000 | 0.3967 | (0.2988, 0.4926) | 0.0000 |
| controlled_vs_candidate_no_context | sentence_score | 0.2000 | (0.1656, 0.2344) | 0.0000 | 0.2000 | (0.1437, 0.2562) | 0.0000 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0438 | (0.0298, 0.0578) | 0.0000 | 0.0438 | (0.0118, 0.0727) | 0.0050 |
| controlled_vs_candidate_no_context | overall_quality | 0.1834 | (0.1703, 0.1966) | 0.0000 | 0.1834 | (0.1582, 0.2049) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2099 | (0.1940, 0.2263) | 0.0000 | 0.2099 | (0.1845, 0.2375) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1731 | (0.1389, 0.2068) | 0.0000 | 0.1731 | (0.1155, 0.2399) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | 0.0355 | (0.0185, 0.0520) | 0.0000 | 0.0355 | (0.0188, 0.0561) | 0.0000 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.2747 | (0.2539, 0.2962) | 0.0000 | 0.2747 | (0.2394, 0.3132) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0587 | (0.0514, 0.0661) | 0.0000 | 0.0587 | (0.0504, 0.0700) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2130 | (0.1702, 0.2560) | 0.0000 | 0.2130 | (0.1439, 0.2882) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | 0.0139 | (-0.0107, 0.0386) | 0.1390 | 0.0139 | (-0.0290, 0.0596) | 0.3010 |
| controlled_vs_baseline_no_context | distinct1 | -0.0331 | (-0.0404, -0.0255) | 1.0000 | -0.0331 | (-0.0451, -0.0190) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | 0.1920 | (0.1176, 0.2661) | 0.0000 | 0.1920 | (0.1045, 0.2813) | 0.0000 |
| controlled_vs_baseline_no_context | sentence_score | 0.1031 | (0.0656, 0.1375) | 0.0000 | 0.1031 | (0.0437, 0.1750) | 0.0000 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0336 | (0.0197, 0.0473) | 0.0000 | 0.0336 | (0.0074, 0.0538) | 0.0077 |
| controlled_vs_baseline_no_context | overall_quality | 0.1528 | (0.1406, 0.1650) | 0.0000 | 0.1528 | (0.1365, 0.1706) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2049 | (0.1884, 0.2226) | 0.0000 | 0.2049 | (0.1772, 0.2358) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1809 | (0.1484, 0.2139) | 0.0000 | 0.1809 | (0.1304, 0.2354) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | 0.0265 | (0.0121, 0.0414) | 0.0000 | 0.0265 | (0.0082, 0.0460) | 0.0030 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.2680 | (0.2460, 0.2898) | 0.0000 | 0.2680 | (0.2292, 0.3097) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0577 | (0.0504, 0.0655) | 0.0000 | 0.0577 | (0.0505, 0.0673) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.2225 | (0.1821, 0.2617) | 0.0000 | 0.2225 | (0.1600, 0.2922) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0144 | (-0.0036, 0.0347) | 0.0610 | 0.0144 | (-0.0126, 0.0475) | 0.1857 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0384 | (-0.0452, -0.0319) | 1.0000 | -0.0384 | (-0.0466, -0.0304) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 0.1610 | (0.1012, 0.2214) | 0.0000 | 0.1610 | (0.0738, 0.2479) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0969 | (0.0625, 0.1313) | 0.0000 | 0.0969 | (0.0406, 0.1500) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0319 | (0.0181, 0.0450) | 0.0000 | 0.0319 | (0.0024, 0.0531) | 0.0153 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1515 | (0.1387, 0.1646) | 0.0000 | 0.1515 | (0.1367, 0.1653) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2099 | (0.1946, 0.2268) | 0.0000 | 0.2099 | (0.1836, 0.2385) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1731 | (0.1394, 0.2062) | 0.0000 | 0.1731 | (0.1184, 0.2369) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0.0355 | (0.0185, 0.0523) | 0.0000 | 0.0355 | (0.0177, 0.0552) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.2747 | (0.2542, 0.2967) | 0.0000 | 0.2747 | (0.2404, 0.3112) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0587 | (0.0513, 0.0666) | 0.0000 | 0.0587 | (0.0501, 0.0698) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.2130 | (0.1706, 0.2532) | 0.0000 | 0.2130 | (0.1421, 0.2850) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0139 | (-0.0104, 0.0404) | 0.1367 | 0.0139 | (-0.0287, 0.0596) | 0.2993 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0331 | (-0.0401, -0.0260) | 1.0000 | -0.0331 | (-0.0455, -0.0188) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.1920 | (0.1152, 0.2658) | 0.0000 | 0.1920 | (0.1062, 0.2824) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.1031 | (0.0656, 0.1375) | 0.0000 | 0.1031 | (0.0437, 0.1719) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0336 | (0.0193, 0.0476) | 0.0000 | 0.0336 | (0.0088, 0.0532) | 0.0037 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1528 | (0.1404, 0.1652) | 0.0000 | 0.1528 | (0.1361, 0.1701) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2049 | (0.1883, 0.2218) | 0.0000 | 0.2049 | (0.1757, 0.2361) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1809 | (0.1490, 0.2145) | 0.0000 | 0.1809 | (0.1318, 0.2353) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | 0.0265 | (0.0124, 0.0405) | 0.0000 | 0.0265 | (0.0075, 0.0453) | 0.0007 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.2680 | (0.2452, 0.2911) | 0.0000 | 0.2680 | (0.2306, 0.3084) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0577 | (0.0504, 0.0653) | 0.0000 | 0.0577 | (0.0506, 0.0675) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.2225 | (0.1828, 0.2622) | 0.0000 | 0.2225 | (0.1597, 0.2882) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0144 | (-0.0033, 0.0349) | 0.0563 | 0.0144 | (-0.0118, 0.0450) | 0.1807 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0384 | (-0.0449, -0.0317) | 1.0000 | -0.0384 | (-0.0463, -0.0305) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 0.1610 | (0.0979, 0.2262) | 0.0000 | 0.1610 | (0.0756, 0.2470) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0969 | (0.0656, 0.1313) | 0.0000 | 0.0969 | (0.0406, 0.1500) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0319 | (0.0188, 0.0454) | 0.0000 | 0.0319 | (0.0037, 0.0531) | 0.0150 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1515 | (0.1388, 0.1646) | 0.0000 | 0.1515 | (0.1378, 0.1656) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 52 | 14 | 46 | 0.6696 | 0.7879 |
| proposed_vs_candidate_no_context | persona_consistency | 24 | 21 | 67 | 0.5134 | 0.5333 |
| proposed_vs_candidate_no_context | naturalness | 33 | 33 | 46 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 40 | 4 | 68 | 0.6607 | 0.9091 |
| proposed_vs_candidate_no_context | context_overlap | 49 | 17 | 46 | 0.6429 | 0.7424 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 15 | 17 | 80 | 0.4911 | 0.4688 |
| proposed_vs_candidate_no_context | persona_style | 11 | 7 | 94 | 0.5179 | 0.6111 |
| proposed_vs_candidate_no_context | distinct1 | 30 | 31 | 51 | 0.4955 | 0.4918 |
| proposed_vs_candidate_no_context | length_score | 27 | 38 | 47 | 0.4509 | 0.4154 |
| proposed_vs_candidate_no_context | sentence_score | 16 | 24 | 72 | 0.4643 | 0.4000 |
| proposed_vs_candidate_no_context | bertscore_f1 | 52 | 24 | 36 | 0.6250 | 0.6842 |
| proposed_vs_candidate_no_context | overall_quality | 46 | 30 | 36 | 0.5714 | 0.6053 |
| proposed_vs_baseline_no_context | context_relevance | 65 | 47 | 0 | 0.5804 | 0.5804 |
| proposed_vs_baseline_no_context | persona_consistency | 25 | 46 | 41 | 0.4062 | 0.3521 |
| proposed_vs_baseline_no_context | naturalness | 20 | 92 | 0 | 0.1786 | 0.1786 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 40 | 25 | 47 | 0.5670 | 0.6154 |
| proposed_vs_baseline_no_context | context_overlap | 71 | 40 | 1 | 0.6384 | 0.6396 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 15 | 31 | 66 | 0.4286 | 0.3261 |
| proposed_vs_baseline_no_context | persona_style | 12 | 26 | 74 | 0.4375 | 0.3158 |
| proposed_vs_baseline_no_context | distinct1 | 17 | 81 | 14 | 0.2143 | 0.1735 |
| proposed_vs_baseline_no_context | length_score | 22 | 89 | 1 | 0.2009 | 0.1982 |
| proposed_vs_baseline_no_context | sentence_score | 13 | 53 | 46 | 0.3214 | 0.1970 |
| proposed_vs_baseline_no_context | bertscore_f1 | 63 | 49 | 0 | 0.5625 | 0.5625 |
| proposed_vs_baseline_no_context | overall_quality | 47 | 65 | 0 | 0.4196 | 0.4196 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 60 | 50 | 2 | 0.5446 | 0.5455 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 23 | 46 | 43 | 0.3973 | 0.3333 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | 20 | 92 | 0 | 0.1786 | 0.1786 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 40 | 25 | 47 | 0.5670 | 0.6154 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 67 | 43 | 2 | 0.6071 | 0.6091 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 16 | 29 | 67 | 0.4420 | 0.3556 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | 10 | 25 | 77 | 0.4330 | 0.2857 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | 10 | 86 | 16 | 0.1607 | 0.1042 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | 21 | 88 | 3 | 0.2009 | 0.1927 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | 8 | 49 | 55 | 0.3170 | 0.1404 |
| proposed_vs_baseline_no_context_phi3_latest | bertscore_f1 | 60 | 52 | 0 | 0.5357 | 0.5357 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 44 | 68 | 0 | 0.3929 | 0.3929 |
| controlled_vs_proposed_raw | context_relevance | 100 | 12 | 0 | 0.8929 | 0.8929 |
| controlled_vs_proposed_raw | persona_consistency | 100 | 6 | 6 | 0.9196 | 0.9434 |
| controlled_vs_proposed_raw | naturalness | 99 | 13 | 0 | 0.8839 | 0.8839 |
| controlled_vs_proposed_raw | context_keyword_coverage | 94 | 6 | 12 | 0.8929 | 0.9400 |
| controlled_vs_proposed_raw | context_overlap | 94 | 18 | 0 | 0.8393 | 0.8393 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 99 | 6 | 7 | 0.9152 | 0.9429 |
| controlled_vs_proposed_raw | persona_style | 28 | 5 | 79 | 0.6027 | 0.8485 |
| controlled_vs_proposed_raw | distinct1 | 66 | 43 | 3 | 0.6027 | 0.6055 |
| controlled_vs_proposed_raw | length_score | 95 | 14 | 3 | 0.8616 | 0.8716 |
| controlled_vs_proposed_raw | sentence_score | 75 | 3 | 34 | 0.8214 | 0.9615 |
| controlled_vs_proposed_raw | bertscore_f1 | 69 | 43 | 0 | 0.6161 | 0.6161 |
| controlled_vs_proposed_raw | overall_quality | 106 | 6 | 0 | 0.9464 | 0.9464 |
| controlled_vs_candidate_no_context | context_relevance | 112 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 97 | 10 | 5 | 0.8884 | 0.9065 |
| controlled_vs_candidate_no_context | naturalness | 95 | 17 | 0 | 0.8482 | 0.8482 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 112 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 110 | 2 | 0 | 0.9821 | 0.9821 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 96 | 10 | 6 | 0.8839 | 0.9057 |
| controlled_vs_candidate_no_context | persona_style | 31 | 5 | 76 | 0.6161 | 0.8611 |
| controlled_vs_candidate_no_context | distinct1 | 61 | 49 | 2 | 0.5536 | 0.5545 |
| controlled_vs_candidate_no_context | length_score | 88 | 20 | 4 | 0.8036 | 0.8148 |
| controlled_vs_candidate_no_context | sentence_score | 65 | 1 | 46 | 0.7857 | 0.9848 |
| controlled_vs_candidate_no_context | bertscore_f1 | 83 | 29 | 0 | 0.7411 | 0.7411 |
| controlled_vs_candidate_no_context | overall_quality | 112 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | context_relevance | 111 | 1 | 0 | 0.9911 | 0.9911 |
| controlled_vs_baseline_no_context | persona_consistency | 90 | 11 | 11 | 0.8527 | 0.8911 |
| controlled_vs_baseline_no_context | naturalness | 79 | 33 | 0 | 0.7054 | 0.7054 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 109 | 0 | 3 | 0.9866 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 106 | 6 | 0 | 0.9464 | 0.9464 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 88 | 8 | 16 | 0.8571 | 0.9167 |
| controlled_vs_baseline_no_context | persona_style | 17 | 12 | 83 | 0.5223 | 0.5862 |
| controlled_vs_baseline_no_context | distinct1 | 20 | 90 | 2 | 0.1875 | 0.1818 |
| controlled_vs_baseline_no_context | length_score | 82 | 25 | 5 | 0.7545 | 0.7664 |
| controlled_vs_baseline_no_context | sentence_score | 39 | 6 | 67 | 0.6473 | 0.8667 |
| controlled_vs_baseline_no_context | bertscore_f1 | 70 | 42 | 0 | 0.6250 | 0.6250 |
| controlled_vs_baseline_no_context | overall_quality | 111 | 1 | 0 | 0.9911 | 0.9911 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 110 | 2 | 0 | 0.9821 | 0.9821 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 93 | 7 | 12 | 0.8839 | 0.9300 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | 69 | 43 | 0 | 0.6161 | 0.6161 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 109 | 1 | 2 | 0.9821 | 0.9909 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 108 | 3 | 1 | 0.9688 | 0.9730 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 91 | 6 | 15 | 0.8795 | 0.9381 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 13 | 7 | 92 | 0.5268 | 0.6500 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | 17 | 94 | 1 | 0.1562 | 0.1532 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 70 | 39 | 3 | 0.6384 | 0.6422 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 36 | 5 | 71 | 0.6384 | 0.8780 |
| controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 74 | 38 | 0 | 0.6607 | 0.6607 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 111 | 1 | 0 | 0.9911 | 0.9911 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 111 | 1 | 0 | 0.9911 | 0.9911 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 90 | 11 | 11 | 0.8527 | 0.8911 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 79 | 33 | 0 | 0.7054 | 0.7054 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 109 | 0 | 3 | 0.9866 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 106 | 6 | 0 | 0.9464 | 0.9464 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 88 | 8 | 16 | 0.8571 | 0.9167 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 17 | 12 | 83 | 0.5223 | 0.5862 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 20 | 90 | 2 | 0.1875 | 0.1818 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 82 | 25 | 5 | 0.7545 | 0.7664 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 39 | 6 | 67 | 0.6473 | 0.8667 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 70 | 42 | 0 | 0.6250 | 0.6250 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 111 | 1 | 0 | 0.9911 | 0.9911 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 110 | 2 | 0 | 0.9821 | 0.9821 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 93 | 7 | 12 | 0.8839 | 0.9300 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | 69 | 43 | 0 | 0.6161 | 0.6161 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 109 | 1 | 2 | 0.9821 | 0.9909 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 108 | 3 | 1 | 0.9688 | 0.9730 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 91 | 6 | 15 | 0.8795 | 0.9381 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 13 | 7 | 92 | 0.5268 | 0.6500 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | 17 | 94 | 1 | 0.1562 | 0.1532 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 70 | 39 | 3 | 0.6384 | 0.6422 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 36 | 5 | 71 | 0.6384 | 0.8780 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 74 | 38 | 0 | 0.6607 | 0.6607 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 111 | 1 | 0 | 0.9911 | 0.9911 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.6964 | 0.0714 | 0.9286 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5179 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.5357 | 0.0000 | 0.0000 |
| baseline_no_context | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline_no_context_phi3_latest | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `8`
- Unique template signatures: `71`
- Template signature ratio: `0.6339`
- Effective sample size by source clustering: `8.00`
- Effective sample size by template-signature clustering: `56.00`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: enabled.

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
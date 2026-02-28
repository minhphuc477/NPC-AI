# Proposal Alignment Evaluation Report

- Run ID: `20260228T083050Z`
- Generated: `2026-02-28T08:31:27.247968+00:00`
- Scenarios: `artifacts\proposal\20260228T083050Z\scenarios.jsonl`
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
| proposed_contextual_controlled | 0.2942 (0.2770, 0.3112) | 0.2276 (0.1979, 0.2585) | 0.8909 (0.8806, 0.9001) | 0.3612 (0.3484, 0.3757) | 0.1182 |
| proposed_contextual | 0.0633 (0.0467, 0.0805) | 0.1525 (0.1289, 0.1778) | 0.8124 (0.7982, 0.8270) | 0.2235 (0.2088, 0.2392) | 0.0697 |
| candidate_no_context | 0.0234 (0.0180, 0.0293) | 0.1331 (0.1114, 0.1570) | 0.8107 (0.7949, 0.8274) | 0.1978 (0.1880, 0.2079) | 0.0445 |
| baseline_no_context | 0.0388 (0.0305, 0.0473) | 0.1655 (0.1456, 0.1860) | 0.8835 (0.8729, 0.8942) | 0.2284 (0.2199, 0.2372) | 0.0587 |
| baseline_no_context_phi3_latest | 0.0488 (0.0400, 0.0580) | 0.1856 (0.1628, 0.2131) | 0.8884 (0.8786, 0.8984) | 0.2398 (0.2301, 0.2499) | 0.0564 |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0398 | 1.7018 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0194 | 0.1460 |
| proposed_vs_candidate_no_context | naturalness | 0.0017 | 0.0021 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0544 | 3.5410 |
| proposed_vs_candidate_no_context | context_overlap | 0.0059 | 0.1408 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0158 | 0.3321 |
| proposed_vs_candidate_no_context | persona_style | 0.0339 | 0.0714 |
| proposed_vs_candidate_no_context | distinct1 | 0.0065 | 0.0070 |
| proposed_vs_candidate_no_context | length_score | -0.0042 | -0.0131 |
| proposed_vs_candidate_no_context | sentence_score | -0.0004 | -0.0006 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0253 | 0.5688 |
| proposed_vs_candidate_no_context | overall_quality | 0.0257 | 0.1298 |
| proposed_vs_baseline_no_context | context_relevance | 0.0245 | 0.6323 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0130 | -0.0784 |
| proposed_vs_baseline_no_context | naturalness | -0.0711 | -0.0805 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0332 | 0.9075 |
| proposed_vs_baseline_no_context | context_overlap | 0.0043 | 0.0975 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 0.0012 | 0.0191 |
| proposed_vs_baseline_no_context | persona_style | -0.0696 | -0.1204 |
| proposed_vs_baseline_no_context | distinct1 | -0.0447 | -0.0454 |
| proposed_vs_baseline_no_context | length_score | -0.2021 | -0.3923 |
| proposed_vs_baseline_no_context | sentence_score | -0.1281 | -0.1480 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0111 | 0.1883 |
| proposed_vs_baseline_no_context | overall_quality | -0.0049 | -0.0215 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.0145 | 0.2974 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | -0.0331 | -0.1782 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0760 | -0.0855 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0185 | 0.3621 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0051 | 0.1180 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | -0.0216 | -0.2544 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | -0.0787 | -0.1340 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0386 | -0.0394 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.2214 | -0.4143 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | -0.1625 | -0.1806 |
| proposed_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0133 | 0.2364 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | -0.0162 | -0.0678 |
| controlled_vs_proposed_raw | context_relevance | 0.2309 | 3.6508 |
| controlled_vs_proposed_raw | persona_consistency | 0.0751 | 0.4926 |
| controlled_vs_proposed_raw | naturalness | 0.0785 | 0.0966 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.3053 | 4.3793 |
| controlled_vs_proposed_raw | context_overlap | 0.0573 | 1.1896 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0754 | 1.1883 |
| controlled_vs_proposed_raw | persona_style | 0.0741 | 0.1456 |
| controlled_vs_proposed_raw | distinct1 | 0.0012 | 0.0013 |
| controlled_vs_proposed_raw | length_score | 0.2619 | 0.8365 |
| controlled_vs_proposed_raw | sentence_score | 0.2562 | 0.3475 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0484 | 0.6942 |
| controlled_vs_proposed_raw | overall_quality | 0.1377 | 0.6161 |
| controlled_vs_candidate_no_context | context_relevance | 0.2708 | 11.5656 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0946 | 0.7105 |
| controlled_vs_candidate_no_context | naturalness | 0.0802 | 0.0990 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3597 | 23.4273 |
| controlled_vs_candidate_no_context | context_overlap | 0.0632 | 1.4980 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0912 | 1.9152 |
| controlled_vs_candidate_no_context | persona_style | 0.1080 | 0.2273 |
| controlled_vs_candidate_no_context | distinct1 | 0.0077 | 0.0083 |
| controlled_vs_candidate_no_context | length_score | 0.2577 | 0.8124 |
| controlled_vs_candidate_no_context | sentence_score | 0.2558 | 0.3466 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0737 | 1.6580 |
| controlled_vs_candidate_no_context | overall_quality | 0.1634 | 0.8259 |
| controlled_vs_baseline_no_context | context_relevance | 0.2554 | 6.5917 |
| controlled_vs_baseline_no_context | persona_consistency | 0.0622 | 0.3756 |
| controlled_vs_baseline_no_context | naturalness | 0.0074 | 0.0084 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3385 | 9.2609 |
| controlled_vs_baseline_no_context | context_overlap | 0.0616 | 1.4031 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.0766 | 1.2302 |
| controlled_vs_baseline_no_context | persona_style | 0.0045 | 0.0077 |
| controlled_vs_baseline_no_context | distinct1 | -0.0435 | -0.0441 |
| controlled_vs_baseline_no_context | length_score | 0.0598 | 0.1161 |
| controlled_vs_baseline_no_context | sentence_score | 0.1281 | 0.1480 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0595 | 1.0133 |
| controlled_vs_baseline_no_context | overall_quality | 0.1328 | 0.5813 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2454 | 5.0337 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.0421 | 0.2267 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | 0.0025 | 0.0029 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3239 | 6.3269 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0624 | 1.4479 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.0537 | 0.6317 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | -0.0047 | -0.0079 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0373 | -0.0382 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 0.0405 | 0.0757 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0938 | 0.1042 |
| controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0618 | 1.0948 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1214 | 0.5065 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2554 | 6.5917 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.0622 | 0.3756 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0.0074 | 0.0084 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3385 | 9.2609 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0616 | 1.4031 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.0766 | 1.2302 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0045 | 0.0077 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0435 | -0.0441 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.0598 | 0.1161 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.1281 | 0.1480 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0595 | 1.0133 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1328 | 0.5813 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2454 | 5.0337 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.0421 | 0.2267 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | 0.0025 | 0.0029 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3239 | 6.3269 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0624 | 1.4479 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.0537 | 0.6317 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | -0.0047 | -0.0079 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0373 | -0.0382 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 0.0405 | 0.0757 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0938 | 0.1042 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0618 | 1.0948 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1214 | 0.5065 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) |
|---|---|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0398 | (0.0224, 0.0585) | 0.0000 |
| proposed_vs_candidate_no_context | persona_consistency | 0.0194 | (-0.0089, 0.0479) | 0.0950 |
| proposed_vs_candidate_no_context | naturalness | 0.0017 | (-0.0154, 0.0194) | 0.4193 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0544 | (0.0317, 0.0782) | 0.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0059 | (0.0000, 0.0122) | 0.0250 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 0.0158 | (-0.0159, 0.0477) | 0.1487 |
| proposed_vs_candidate_no_context | persona_style | 0.0339 | (0.0077, 0.0626) | 0.0033 |
| proposed_vs_candidate_no_context | distinct1 | 0.0065 | (-0.0010, 0.0139) | 0.0460 |
| proposed_vs_candidate_no_context | length_score | -0.0042 | (-0.0705, 0.0622) | 0.5520 |
| proposed_vs_candidate_no_context | sentence_score | -0.0004 | (-0.0344, 0.0339) | 0.5323 |
| proposed_vs_candidate_no_context | bertscore_f1 | 0.0253 | (0.0141, 0.0368) | 0.0000 |
| proposed_vs_candidate_no_context | overall_quality | 0.0257 | (0.0097, 0.0422) | 0.0010 |
| proposed_vs_baseline_no_context | context_relevance | 0.0245 | (0.0053, 0.0454) | 0.0063 |
| proposed_vs_baseline_no_context | persona_consistency | -0.0130 | (-0.0395, 0.0132) | 0.8390 |
| proposed_vs_baseline_no_context | naturalness | -0.0711 | (-0.0878, -0.0540) | 1.0000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0332 | (0.0087, 0.0598) | 0.0043 |
| proposed_vs_baseline_no_context | context_overlap | 0.0043 | (-0.0034, 0.0125) | 0.1410 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 0.0012 | (-0.0290, 0.0321) | 0.4730 |
| proposed_vs_baseline_no_context | persona_style | -0.0696 | (-0.1134, -0.0312) | 0.9997 |
| proposed_vs_baseline_no_context | distinct1 | -0.0447 | (-0.0522, -0.0366) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | -0.2021 | (-0.2714, -0.1336) | 1.0000 |
| proposed_vs_baseline_no_context | sentence_score | -0.1281 | (-0.1687, -0.0844) | 1.0000 |
| proposed_vs_baseline_no_context | bertscore_f1 | 0.0111 | (-0.0016, 0.0238) | 0.0457 |
| proposed_vs_baseline_no_context | overall_quality | -0.0049 | (-0.0206, 0.0124) | 0.7327 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.0145 | (-0.0043, 0.0338) | 0.0667 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | -0.0331 | (-0.0660, 0.0005) | 0.9740 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0760 | (-0.0916, -0.0601) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0185 | (-0.0063, 0.0460) | 0.0740 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0051 | (-0.0013, 0.0119) | 0.0607 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | -0.0216 | (-0.0606, 0.0140) | 0.8777 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | -0.0787 | (-0.1242, -0.0381) | 0.9997 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0386 | (-0.0465, -0.0303) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.2214 | (-0.2818, -0.1592) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | -0.1625 | (-0.2000, -0.1219) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0133 | (0.0006, 0.0257) | 0.0193 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | -0.0162 | (-0.0337, 0.0014) | 0.9640 |
| controlled_vs_proposed_raw | context_relevance | 0.2309 | (0.2065, 0.2550) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.0751 | (0.0443, 0.1049) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0785 | (0.0617, 0.0941) | 0.0000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.3053 | (0.2716, 0.3361) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0573 | (0.0472, 0.0674) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0754 | (0.0406, 0.1113) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0741 | (0.0357, 0.1130) | 0.0000 |
| controlled_vs_proposed_raw | distinct1 | 0.0012 | (-0.0066, 0.0090) | 0.3763 |
| controlled_vs_proposed_raw | length_score | 0.2619 | (0.1967, 0.3274) | 0.0000 |
| controlled_vs_proposed_raw | sentence_score | 0.2562 | (0.2250, 0.2844) | 0.0000 |
| controlled_vs_proposed_raw | bertscore_f1 | 0.0484 | (0.0337, 0.0630) | 0.0000 |
| controlled_vs_proposed_raw | overall_quality | 0.1377 | (0.1200, 0.1557) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2708 | (0.2535, 0.2901) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0946 | (0.0625, 0.1265) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0802 | (0.0605, 0.0986) | 0.0000 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3597 | (0.3358, 0.3839) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0632 | (0.0557, 0.0716) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0912 | (0.0543, 0.1306) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.1080 | (0.0662, 0.1527) | 0.0000 |
| controlled_vs_candidate_no_context | distinct1 | 0.0077 | (-0.0005, 0.0158) | 0.0333 |
| controlled_vs_candidate_no_context | length_score | 0.2577 | (0.1783, 0.3351) | 0.0000 |
| controlled_vs_candidate_no_context | sentence_score | 0.2558 | (0.2250, 0.2862) | 0.0000 |
| controlled_vs_candidate_no_context | bertscore_f1 | 0.0737 | (0.0596, 0.0869) | 0.0000 |
| controlled_vs_candidate_no_context | overall_quality | 0.1634 | (0.1497, 0.1778) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2554 | (0.2373, 0.2746) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.0622 | (0.0326, 0.0941) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | 0.0074 | (-0.0068, 0.0212) | 0.1497 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3385 | (0.3144, 0.3645) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0616 | (0.0528, 0.0702) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.0766 | (0.0400, 0.1153) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | 0.0045 | (-0.0148, 0.0244) | 0.3250 |
| controlled_vs_baseline_no_context | distinct1 | -0.0435 | (-0.0494, -0.0374) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | 0.0598 | (-0.0054, 0.1232) | 0.0340 |
| controlled_vs_baseline_no_context | sentence_score | 0.1281 | (0.0969, 0.1625) | 0.0000 |
| controlled_vs_baseline_no_context | bertscore_f1 | 0.0595 | (0.0441, 0.0735) | 0.0000 |
| controlled_vs_baseline_no_context | overall_quality | 0.1328 | (0.1200, 0.1454) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2454 | (0.2280, 0.2630) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.0421 | (0.0099, 0.0744) | 0.0083 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | 0.0025 | (-0.0105, 0.0151) | 0.3447 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3239 | (0.3002, 0.3478) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0624 | (0.0538, 0.0710) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.0537 | (0.0136, 0.0938) | 0.0033 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | -0.0047 | (-0.0265, 0.0172) | 0.6660 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0373 | (-0.0436, -0.0309) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 0.0405 | (-0.0208, 0.0991) | 0.0910 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0938 | (0.0656, 0.1250) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0618 | (0.0478, 0.0756) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1214 | (0.1077, 0.1354) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2554 | (0.2374, 0.2745) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.0622 | (0.0307, 0.0922) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0.0074 | (-0.0076, 0.0210) | 0.1537 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3385 | (0.3137, 0.3632) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0616 | (0.0528, 0.0705) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.0766 | (0.0396, 0.1163) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 0.0045 | (-0.0144, 0.0231) | 0.3353 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0435 | (-0.0492, -0.0375) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.0598 | (-0.0057, 0.1214) | 0.0407 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.1281 | (0.0969, 0.1594) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 0.0595 | (0.0442, 0.0738) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1328 | (0.1203, 0.1461) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2454 | (0.2279, 0.2629) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.0421 | (0.0092, 0.0741) | 0.0067 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | 0.0025 | (-0.0100, 0.0152) | 0.3360 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3239 | (0.3008, 0.3472) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0624 | (0.0540, 0.0710) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.0537 | (0.0139, 0.0939) | 0.0030 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | -0.0047 | (-0.0261, 0.0171) | 0.6500 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0373 | (-0.0438, -0.0310) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 0.0405 | (-0.0185, 0.1000) | 0.0890 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0938 | (0.0656, 0.1250) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 0.0618 | (0.0472, 0.0760) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1214 | (0.1076, 0.1348) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 37 | 26 | 49 | 0.5491 | 0.5873 |
| proposed_vs_candidate_no_context | persona_consistency | 24 | 15 | 73 | 0.5402 | 0.6154 |
| proposed_vs_candidate_no_context | naturalness | 32 | 30 | 50 | 0.5089 | 0.5161 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 32 | 8 | 72 | 0.6071 | 0.8000 |
| proposed_vs_candidate_no_context | context_overlap | 34 | 29 | 49 | 0.5223 | 0.5397 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 17 | 12 | 83 | 0.5223 | 0.5862 |
| proposed_vs_candidate_no_context | persona_style | 15 | 6 | 91 | 0.5402 | 0.7143 |
| proposed_vs_candidate_no_context | distinct1 | 35 | 21 | 56 | 0.5625 | 0.6250 |
| proposed_vs_candidate_no_context | length_score | 30 | 30 | 52 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | sentence_score | 16 | 16 | 80 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | bertscore_f1 | 55 | 24 | 33 | 0.6384 | 0.6962 |
| proposed_vs_candidate_no_context | overall_quality | 53 | 26 | 33 | 0.6205 | 0.6709 |
| proposed_vs_baseline_no_context | context_relevance | 54 | 58 | 0 | 0.4821 | 0.4821 |
| proposed_vs_baseline_no_context | persona_consistency | 21 | 47 | 44 | 0.3839 | 0.3088 |
| proposed_vs_baseline_no_context | naturalness | 32 | 79 | 1 | 0.2902 | 0.2883 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 30 | 25 | 57 | 0.5223 | 0.5455 |
| proposed_vs_baseline_no_context | context_overlap | 57 | 55 | 0 | 0.5089 | 0.5089 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 16 | 23 | 73 | 0.4688 | 0.4103 |
| proposed_vs_baseline_no_context | persona_style | 10 | 29 | 73 | 0.4152 | 0.2564 |
| proposed_vs_baseline_no_context | distinct1 | 14 | 82 | 16 | 0.1964 | 0.1458 |
| proposed_vs_baseline_no_context | length_score | 33 | 74 | 5 | 0.3170 | 0.3084 |
| proposed_vs_baseline_no_context | sentence_score | 11 | 52 | 49 | 0.3170 | 0.1746 |
| proposed_vs_baseline_no_context | bertscore_f1 | 65 | 47 | 0 | 0.5804 | 0.5804 |
| proposed_vs_baseline_no_context | overall_quality | 38 | 74 | 0 | 0.3393 | 0.3393 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 50 | 62 | 0 | 0.4464 | 0.4464 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 19 | 48 | 45 | 0.3705 | 0.2836 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | 21 | 90 | 1 | 0.1920 | 0.1892 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 30 | 32 | 50 | 0.4911 | 0.4839 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 57 | 55 | 0 | 0.5089 | 0.5089 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 16 | 33 | 63 | 0.4241 | 0.3265 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | 8 | 27 | 77 | 0.4152 | 0.2286 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | 19 | 80 | 13 | 0.2277 | 0.1919 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | 22 | 85 | 5 | 0.2188 | 0.2056 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | 6 | 58 | 48 | 0.2679 | 0.0938 |
| proposed_vs_baseline_no_context_phi3_latest | bertscore_f1 | 69 | 43 | 0 | 0.6161 | 0.6161 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 36 | 76 | 0 | 0.3214 | 0.3214 |
| controlled_vs_proposed_raw | context_relevance | 106 | 6 | 0 | 0.9464 | 0.9464 |
| controlled_vs_proposed_raw | persona_consistency | 60 | 12 | 40 | 0.7143 | 0.8333 |
| controlled_vs_proposed_raw | naturalness | 91 | 21 | 0 | 0.8125 | 0.8125 |
| controlled_vs_proposed_raw | context_keyword_coverage | 105 | 3 | 4 | 0.9554 | 0.9722 |
| controlled_vs_proposed_raw | context_overlap | 101 | 11 | 0 | 0.9018 | 0.9018 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 43 | 12 | 57 | 0.6384 | 0.7818 |
| controlled_vs_proposed_raw | persona_style | 26 | 10 | 76 | 0.5714 | 0.7222 |
| controlled_vs_proposed_raw | distinct1 | 65 | 46 | 1 | 0.5848 | 0.5856 |
| controlled_vs_proposed_raw | length_score | 78 | 33 | 1 | 0.7009 | 0.7027 |
| controlled_vs_proposed_raw | sentence_score | 82 | 0 | 30 | 0.8661 | 1.0000 |
| controlled_vs_proposed_raw | bertscore_f1 | 80 | 32 | 0 | 0.7143 | 0.7143 |
| controlled_vs_proposed_raw | overall_quality | 102 | 10 | 0 | 0.9107 | 0.9107 |
| controlled_vs_candidate_no_context | context_relevance | 112 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 66 | 9 | 37 | 0.7545 | 0.8800 |
| controlled_vs_candidate_no_context | naturalness | 85 | 27 | 0 | 0.7589 | 0.7589 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 112 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 107 | 4 | 1 | 0.9598 | 0.9640 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 45 | 9 | 58 | 0.6607 | 0.8333 |
| controlled_vs_candidate_no_context | persona_style | 32 | 9 | 71 | 0.6027 | 0.7805 |
| controlled_vs_candidate_no_context | distinct1 | 75 | 36 | 1 | 0.6741 | 0.6757 |
| controlled_vs_candidate_no_context | length_score | 80 | 32 | 0 | 0.7143 | 0.7143 |
| controlled_vs_candidate_no_context | sentence_score | 81 | 0 | 31 | 0.8616 | 1.0000 |
| controlled_vs_candidate_no_context | bertscore_f1 | 94 | 18 | 0 | 0.8393 | 0.8393 |
| controlled_vs_candidate_no_context | overall_quality | 111 | 1 | 0 | 0.9911 | 0.9911 |
| controlled_vs_baseline_no_context | context_relevance | 112 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 47 | 21 | 44 | 0.6161 | 0.6912 |
| controlled_vs_baseline_no_context | naturalness | 60 | 51 | 1 | 0.5402 | 0.5405 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 110 | 0 | 2 | 0.9911 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 104 | 8 | 0 | 0.9286 | 0.9286 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 39 | 16 | 57 | 0.6027 | 0.7091 |
| controlled_vs_baseline_no_context | persona_style | 15 | 15 | 82 | 0.5000 | 0.5000 |
| controlled_vs_baseline_no_context | distinct1 | 10 | 99 | 3 | 0.1027 | 0.0917 |
| controlled_vs_baseline_no_context | length_score | 63 | 47 | 2 | 0.5714 | 0.5727 |
| controlled_vs_baseline_no_context | sentence_score | 41 | 0 | 71 | 0.6830 | 1.0000 |
| controlled_vs_baseline_no_context | bertscore_f1 | 89 | 23 | 0 | 0.7946 | 0.7946 |
| controlled_vs_baseline_no_context | overall_quality | 109 | 3 | 0 | 0.9732 | 0.9732 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 112 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 45 | 22 | 45 | 0.6027 | 0.6716 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | 55 | 55 | 2 | 0.5000 | 0.5000 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 111 | 0 | 1 | 0.9955 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 107 | 5 | 0 | 0.9554 | 0.9554 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 40 | 19 | 53 | 0.5938 | 0.6780 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 11 | 15 | 86 | 0.4821 | 0.4231 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | 15 | 93 | 4 | 0.1518 | 0.1389 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 58 | 47 | 7 | 0.5491 | 0.5524 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 31 | 1 | 80 | 0.6339 | 0.9688 |
| controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 91 | 21 | 0 | 0.8125 | 0.8125 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 108 | 4 | 0 | 0.9643 | 0.9643 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 112 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 47 | 21 | 44 | 0.6161 | 0.6912 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 60 | 51 | 1 | 0.5402 | 0.5405 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 110 | 0 | 2 | 0.9911 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 104 | 8 | 0 | 0.9286 | 0.9286 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 39 | 16 | 57 | 0.6027 | 0.7091 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 15 | 15 | 82 | 0.5000 | 0.5000 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 10 | 99 | 3 | 0.1027 | 0.0917 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 63 | 47 | 2 | 0.5714 | 0.5727 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 41 | 0 | 71 | 0.6830 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | bertscore_f1 | 89 | 23 | 0 | 0.7946 | 0.7946 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 109 | 3 | 0 | 0.9732 | 0.9732 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 112 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 45 | 22 | 45 | 0.6027 | 0.6716 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | 55 | 55 | 2 | 0.5000 | 0.5000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 111 | 0 | 1 | 0.9955 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 107 | 5 | 0 | 0.9554 | 0.9554 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 40 | 19 | 53 | 0.5938 | 0.6780 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 11 | 15 | 86 | 0.4821 | 0.4231 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | 15 | 93 | 4 | 0.1518 | 0.1389 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 58 | 47 | 7 | 0.5491 | 0.5524 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 31 | 1 | 80 | 0.6339 | 0.9688 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | bertscore_f1 | 91 | 21 | 0 | 0.8125 | 0.8125 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 108 | 4 | 0 | 0.9643 | 0.9643 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

- BERTScore status: enabled.

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
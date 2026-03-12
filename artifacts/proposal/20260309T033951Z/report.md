# Proposal Alignment Evaluation Report

- Run ID: `20260309T033951Z`
- Generated: `2026-03-09T03:40:46.252633+00:00`
- Scenarios: `artifacts\proposal\20260309T033951Z\scenarios.jsonl`
- Scenario count: `144`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off
- `baseline_no_context`: model `phi3:mini`, dynamic_context=off, response_control=off
- `baseline_no_context_phi3_latest`: model `phi3:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2683 (0.2548, 0.2831) | 0.3012 (0.2824, 0.3224) | 0.8951 (0.8844, 0.9048) | 0.4072 (0.3975, 0.4158) | n/a |
| proposed_contextual | 0.0928 (0.0735, 0.1122) | 0.1543 (0.1350, 0.1756) | 0.8167 (0.8043, 0.8297) | 0.2551 (0.2421, 0.2702) | n/a |
| candidate_no_context | 0.0242 (0.0188, 0.0302) | 0.1605 (0.1409, 0.1831) | 0.8264 (0.8140, 0.8386) | 0.2279 (0.2190, 0.2379) | n/a |
| baseline_no_context | 0.0371 (0.0300, 0.0448) | 0.1533 (0.1393, 0.1678) | 0.8874 (0.8786, 0.8961) | 0.2429 (0.2353, 0.2510) | n/a |
| baseline_no_context_phi3_latest | 0.0341 (0.0273, 0.0413) | 0.1541 (0.1386, 0.1699) | 0.8883 (0.8788, 0.8977) | 0.2421 (0.2343, 0.2504) | n/a |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0685 | 2.8284 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0062 | -0.0388 |
| proposed_vs_candidate_no_context | naturalness | -0.0098 | -0.0118 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0886 | 4.5037 |
| proposed_vs_candidate_no_context | context_overlap | 0.0217 | 0.6227 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0119 | -0.1707 |
| proposed_vs_candidate_no_context | persona_style | 0.0165 | 0.0315 |
| proposed_vs_candidate_no_context | distinct1 | -0.0041 | -0.0043 |
| proposed_vs_candidate_no_context | length_score | -0.0273 | -0.0802 |
| proposed_vs_candidate_no_context | sentence_score | -0.0292 | -0.0360 |
| proposed_vs_candidate_no_context | overall_quality | 0.0272 | 0.1195 |
| proposed_vs_baseline_no_context | context_relevance | 0.0557 | 1.4992 |
| proposed_vs_baseline_no_context | persona_consistency | 0.0010 | 0.0062 |
| proposed_vs_baseline_no_context | naturalness | -0.0707 | -0.0797 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0711 | 1.9115 |
| proposed_vs_baseline_no_context | context_overlap | 0.0196 | 0.5312 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 0.0069 | 0.1350 |
| proposed_vs_baseline_no_context | persona_style | -0.0227 | -0.0404 |
| proposed_vs_baseline_no_context | distinct1 | -0.0392 | -0.0401 |
| proposed_vs_baseline_no_context | length_score | -0.2375 | -0.4311 |
| proposed_vs_baseline_no_context | sentence_score | -0.0753 | -0.0880 |
| proposed_vs_baseline_no_context | overall_quality | 0.0123 | 0.0506 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.0586 | 1.7169 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 0.0001 | 0.0007 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0716 | -0.0806 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0752 | 2.2704 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0200 | 0.5474 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.0031 | 0.0574 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | -0.0120 | -0.0218 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0381 | -0.0390 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.2306 | -0.4238 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | -0.1021 | -0.1156 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 0.0130 | 0.0538 |
| controlled_vs_proposed_raw | context_relevance | 0.1755 | 1.8915 |
| controlled_vs_proposed_raw | persona_consistency | 0.1469 | 0.9524 |
| controlled_vs_proposed_raw | naturalness | 0.0785 | 0.0961 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2301 | 2.1246 |
| controlled_vs_proposed_raw | context_overlap | 0.0481 | 0.8509 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1785 | 3.0858 |
| controlled_vs_proposed_raw | persona_style | 0.0206 | 0.0382 |
| controlled_vs_proposed_raw | distinct1 | -0.0091 | -0.0097 |
| controlled_vs_proposed_raw | length_score | 0.3398 | 1.0842 |
| controlled_vs_proposed_raw | sentence_score | 0.1434 | 0.1836 |
| controlled_vs_proposed_raw | overall_quality | 0.1521 | 0.5960 |
| controlled_vs_candidate_no_context | context_relevance | 0.2440 | 10.0700 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1407 | 0.8766 |
| controlled_vs_candidate_no_context | naturalness | 0.0687 | 0.0831 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3187 | 16.1968 |
| controlled_vs_candidate_no_context | context_overlap | 0.0699 | 2.0035 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1666 | 2.3883 |
| controlled_vs_candidate_no_context | persona_style | 0.0371 | 0.0709 |
| controlled_vs_candidate_no_context | distinct1 | -0.0132 | -0.0139 |
| controlled_vs_candidate_no_context | length_score | 0.3125 | 0.9171 |
| controlled_vs_candidate_no_context | sentence_score | 0.1142 | 0.1410 |
| controlled_vs_candidate_no_context | overall_quality | 0.1793 | 0.7866 |
| controlled_vs_baseline_no_context | context_relevance | 0.2312 | 6.2266 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1479 | 0.9646 |
| controlled_vs_baseline_no_context | naturalness | 0.0077 | 0.0087 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3012 | 8.0970 |
| controlled_vs_baseline_no_context | context_overlap | 0.0678 | 1.8340 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1854 | 3.6372 |
| controlled_vs_baseline_no_context | persona_style | -0.0021 | -0.0037 |
| controlled_vs_baseline_no_context | distinct1 | -0.0483 | -0.0493 |
| controlled_vs_baseline_no_context | length_score | 0.1023 | 0.1857 |
| controlled_vs_baseline_no_context | sentence_score | 0.0681 | 0.0794 |
| controlled_vs_baseline_no_context | overall_quality | 0.1643 | 0.6766 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2341 | 6.8561 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1470 | 0.9537 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | 0.0069 | 0.0078 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3053 | 9.2186 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0682 | 1.8642 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1816 | 3.3204 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0086 | 0.0155 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0472 | -0.0483 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 0.1093 | 0.2009 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0413 | 0.0468 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1651 | 0.6818 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2312 | 6.2266 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1479 | 0.9646 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0.0077 | 0.0087 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3012 | 8.0970 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0678 | 1.8340 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1854 | 3.6372 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | -0.0021 | -0.0037 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0483 | -0.0493 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.1023 | 0.1857 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0681 | 0.0794 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1643 | 0.6766 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2341 | 6.8561 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1470 | 0.9537 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | 0.0069 | 0.0078 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3053 | 9.2186 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0682 | 1.8642 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1816 | 3.3204 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0086 | 0.0155 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0472 | -0.0483 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 0.1093 | 0.2009 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0413 | 0.0468 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1651 | 0.6818 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0685 | (0.0507, 0.0865) | 0.0000 | 0.0685 | (0.0363, 0.1043) | 0.0000 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0062 | (-0.0245, 0.0124) | 0.7450 | -0.0062 | (-0.0246, 0.0089) | 0.7557 |
| proposed_vs_candidate_no_context | naturalness | -0.0098 | (-0.0234, 0.0044) | 0.9190 | -0.0098 | (-0.0276, 0.0051) | 0.8830 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0886 | (0.0645, 0.1147) | 0.0000 | 0.0886 | (0.0441, 0.1347) | 0.0000 |
| proposed_vs_candidate_no_context | context_overlap | 0.0217 | (0.0146, 0.0290) | 0.0000 | 0.0217 | (0.0094, 0.0341) | 0.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0119 | (-0.0355, 0.0112) | 0.8460 | -0.0119 | (-0.0344, 0.0079) | 0.8670 |
| proposed_vs_candidate_no_context | persona_style | 0.0165 | (-0.0015, 0.0358) | 0.0413 | 0.0165 | (-0.0012, 0.0383) | 0.0390 |
| proposed_vs_candidate_no_context | distinct1 | -0.0041 | (-0.0114, 0.0032) | 0.8690 | -0.0041 | (-0.0115, 0.0029) | 0.8583 |
| proposed_vs_candidate_no_context | length_score | -0.0273 | (-0.0792, 0.0227) | 0.8527 | -0.0273 | (-0.0882, 0.0225) | 0.8147 |
| proposed_vs_candidate_no_context | sentence_score | -0.0292 | (-0.0632, 0.0049) | 0.9590 | -0.0292 | (-0.0656, 0.0073) | 0.9480 |
| proposed_vs_candidate_no_context | overall_quality | 0.0272 | (0.0137, 0.0408) | 0.0003 | 0.0272 | (0.0067, 0.0494) | 0.0020 |
| proposed_vs_baseline_no_context | context_relevance | 0.0557 | (0.0374, 0.0753) | 0.0000 | 0.0557 | (0.0180, 0.0962) | 0.0007 |
| proposed_vs_baseline_no_context | persona_consistency | 0.0010 | (-0.0211, 0.0244) | 0.4850 | 0.0010 | (-0.0344, 0.0378) | 0.5020 |
| proposed_vs_baseline_no_context | naturalness | -0.0707 | (-0.0849, -0.0569) | 1.0000 | -0.0707 | (-0.0932, -0.0508) | 1.0000 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 0.0711 | (0.0465, 0.0975) | 0.0000 | 0.0711 | (0.0242, 0.1224) | 0.0003 |
| proposed_vs_baseline_no_context | context_overlap | 0.0196 | (0.0118, 0.0280) | 0.0000 | 0.0196 | (0.0072, 0.0310) | 0.0000 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 0.0069 | (-0.0198, 0.0341) | 0.3267 | 0.0069 | (-0.0298, 0.0486) | 0.3687 |
| proposed_vs_baseline_no_context | persona_style | -0.0227 | (-0.0464, -0.0018) | 0.9840 | -0.0227 | (-0.0671, 0.0166) | 0.8447 |
| proposed_vs_baseline_no_context | distinct1 | -0.0392 | (-0.0466, -0.0313) | 1.0000 | -0.0392 | (-0.0531, -0.0261) | 1.0000 |
| proposed_vs_baseline_no_context | length_score | -0.2375 | (-0.2873, -0.1868) | 1.0000 | -0.2375 | (-0.3088, -0.1757) | 1.0000 |
| proposed_vs_baseline_no_context | sentence_score | -0.0753 | (-0.1118, -0.0365) | 1.0000 | -0.0753 | (-0.1215, -0.0267) | 1.0000 |
| proposed_vs_baseline_no_context | overall_quality | 0.0123 | (-0.0046, 0.0283) | 0.0643 | 0.0123 | (-0.0175, 0.0416) | 0.2147 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 0.0586 | (0.0386, 0.0802) | 0.0000 | 0.0586 | (0.0179, 0.0991) | 0.0010 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 0.0001 | (-0.0210, 0.0224) | 0.5050 | 0.0001 | (-0.0291, 0.0311) | 0.4887 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | -0.0716 | (-0.0880, -0.0546) | 1.0000 | -0.0716 | (-0.0988, -0.0436) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.0752 | (0.0489, 0.1009) | 0.0000 | 0.0752 | (0.0217, 0.1275) | 0.0010 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 0.0200 | (0.0124, 0.0277) | 0.0000 | 0.0200 | (0.0067, 0.0328) | 0.0010 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.0031 | (-0.0224, 0.0305) | 0.4243 | 0.0031 | (-0.0351, 0.0395) | 0.4420 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | -0.0120 | (-0.0345, 0.0065) | 0.8833 | -0.0120 | (-0.0485, 0.0207) | 0.7330 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | -0.0381 | (-0.0461, -0.0301) | 1.0000 | -0.0381 | (-0.0497, -0.0268) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | -0.2306 | (-0.2914, -0.1724) | 1.0000 | -0.2306 | (-0.3209, -0.1417) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | -0.1021 | (-0.1410, -0.0632) | 1.0000 | -0.1021 | (-0.1580, -0.0486) | 1.0000 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 0.0130 | (-0.0024, 0.0295) | 0.0580 | 0.0130 | (-0.0206, 0.0443) | 0.2213 |
| controlled_vs_proposed_raw | context_relevance | 0.1755 | (0.1551, 0.1963) | 0.0000 | 0.1755 | (0.1349, 0.2176) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.1469 | (0.1248, 0.1703) | 0.0000 | 0.1469 | (0.1137, 0.1762) | 0.0000 |
| controlled_vs_proposed_raw | naturalness | 0.0785 | (0.0609, 0.0960) | 0.0000 | 0.0785 | (0.0410, 0.1161) | 0.0000 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2301 | (0.2016, 0.2586) | 0.0000 | 0.2301 | (0.1761, 0.2842) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0481 | (0.0392, 0.0567) | 0.0000 | 0.0481 | (0.0359, 0.0599) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.1785 | (0.1493, 0.2047) | 0.0000 | 0.1785 | (0.1377, 0.2165) | 0.0000 |
| controlled_vs_proposed_raw | persona_style | 0.0206 | (0.0034, 0.0409) | 0.0073 | 0.0206 | (0.0010, 0.0496) | 0.0193 |
| controlled_vs_proposed_raw | distinct1 | -0.0091 | (-0.0168, -0.0012) | 0.9873 | -0.0091 | (-0.0239, 0.0089) | 0.8493 |
| controlled_vs_proposed_raw | length_score | 0.3398 | (0.2688, 0.4086) | 0.0000 | 0.3398 | (0.2076, 0.4720) | 0.0000 |
| controlled_vs_proposed_raw | sentence_score | 0.1434 | (0.1069, 0.1799) | 0.0000 | 0.1434 | (0.0851, 0.1993) | 0.0000 |
| controlled_vs_proposed_raw | overall_quality | 0.1521 | (0.1363, 0.1676) | 0.0000 | 0.1521 | (0.1239, 0.1823) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2440 | (0.2293, 0.2582) | 0.0000 | 0.2440 | (0.2200, 0.2677) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.1407 | (0.1190, 0.1622) | 0.0000 | 0.1407 | (0.1100, 0.1690) | 0.0000 |
| controlled_vs_candidate_no_context | naturalness | 0.0687 | (0.0492, 0.0870) | 0.0000 | 0.0687 | (0.0294, 0.1097) | 0.0003 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3187 | (0.2996, 0.3379) | 0.0000 | 0.3187 | (0.2867, 0.3501) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0699 | (0.0639, 0.0762) | 0.0000 | 0.0699 | (0.0626, 0.0763) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.1666 | (0.1419, 0.1914) | 0.0000 | 0.1666 | (0.1305, 0.2009) | 0.0000 |
| controlled_vs_candidate_no_context | persona_style | 0.0371 | (0.0146, 0.0618) | 0.0013 | 0.0371 | (0.0000, 0.0853) | 0.0263 |
| controlled_vs_candidate_no_context | distinct1 | -0.0132 | (-0.0211, -0.0055) | 1.0000 | -0.0132 | (-0.0297, 0.0042) | 0.9363 |
| controlled_vs_candidate_no_context | length_score | 0.3125 | (0.2370, 0.3840) | 0.0000 | 0.3125 | (0.1803, 0.4576) | 0.0000 |
| controlled_vs_candidate_no_context | sentence_score | 0.1142 | (0.0778, 0.1531) | 0.0000 | 0.1142 | (0.0486, 0.1751) | 0.0003 |
| controlled_vs_candidate_no_context | overall_quality | 0.1793 | (0.1684, 0.1911) | 0.0000 | 0.1793 | (0.1570, 0.2009) | 0.0000 |
| controlled_vs_baseline_no_context | context_relevance | 0.2312 | (0.2171, 0.2457) | 0.0000 | 0.2312 | (0.2122, 0.2479) | 0.0000 |
| controlled_vs_baseline_no_context | persona_consistency | 0.1479 | (0.1261, 0.1685) | 0.0000 | 0.1479 | (0.1091, 0.1866) | 0.0000 |
| controlled_vs_baseline_no_context | naturalness | 0.0077 | (-0.0058, 0.0201) | 0.1257 | 0.0077 | (-0.0117, 0.0265) | 0.2200 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 0.3012 | (0.2820, 0.3220) | 0.0000 | 0.3012 | (0.2746, 0.3245) | 0.0000 |
| controlled_vs_baseline_no_context | context_overlap | 0.0678 | (0.0620, 0.0739) | 0.0000 | 0.0678 | (0.0609, 0.0757) | 0.0000 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1854 | (0.1598, 0.2103) | 0.0000 | 0.1854 | (0.1398, 0.2309) | 0.0000 |
| controlled_vs_baseline_no_context | persona_style | -0.0021 | (-0.0208, 0.0151) | 0.5737 | -0.0021 | (-0.0347, 0.0215) | 0.5400 |
| controlled_vs_baseline_no_context | distinct1 | -0.0483 | (-0.0552, -0.0413) | 1.0000 | -0.0483 | (-0.0561, -0.0404) | 1.0000 |
| controlled_vs_baseline_no_context | length_score | 0.1023 | (0.0467, 0.1586) | 0.0007 | 0.1023 | (0.0160, 0.1831) | 0.0077 |
| controlled_vs_baseline_no_context | sentence_score | 0.0681 | (0.0340, 0.1045) | 0.0000 | 0.0681 | (0.0194, 0.1142) | 0.0017 |
| controlled_vs_baseline_no_context | overall_quality | 0.1643 | (0.1538, 0.1751) | 0.0000 | 0.1643 | (0.1523, 0.1760) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2341 | (0.2192, 0.2493) | 0.0000 | 0.2341 | (0.2159, 0.2499) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1470 | (0.1267, 0.1684) | 0.0000 | 0.1470 | (0.1040, 0.1778) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | 0.0069 | (-0.0082, 0.0212) | 0.1813 | 0.0069 | (-0.0047, 0.0192) | 0.1297 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3053 | (0.2853, 0.3247) | 0.0000 | 0.3053 | (0.2794, 0.3274) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0682 | (0.0617, 0.0748) | 0.0000 | 0.0682 | (0.0596, 0.0779) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1816 | (0.1562, 0.2082) | 0.0000 | 0.1816 | (0.1318, 0.2186) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0086 | (-0.0039, 0.0215) | 0.0900 | 0.0086 | (-0.0062, 0.0254) | 0.1707 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0472 | (-0.0547, -0.0396) | 1.0000 | -0.0472 | (-0.0547, -0.0390) | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 0.1093 | (0.0479, 0.1720) | 0.0000 | 0.1093 | (0.0556, 0.1660) | 0.0000 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0413 | (0.0073, 0.0753) | 0.0120 | 0.0413 | (-0.0024, 0.0729) | 0.0420 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1651 | (0.1544, 0.1757) | 0.0000 | 0.1651 | (0.1502, 0.1770) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 0.2312 | (0.2164, 0.2456) | 0.0000 | 0.2312 | (0.2118, 0.2482) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 0.1479 | (0.1271, 0.1686) | 0.0000 | 0.1479 | (0.1108, 0.1862) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 0.0077 | (-0.0055, 0.0205) | 0.1250 | 0.0077 | (-0.0123, 0.0262) | 0.2153 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 0.3012 | (0.2813, 0.3208) | 0.0000 | 0.3012 | (0.2740, 0.3246) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 0.0678 | (0.0620, 0.0738) | 0.0000 | 0.0678 | (0.0608, 0.0764) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 0.1854 | (0.1617, 0.2110) | 0.0000 | 0.1854 | (0.1419, 0.2309) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | -0.0021 | (-0.0199, 0.0147) | 0.5813 | -0.0021 | (-0.0347, 0.0247) | 0.5473 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | -0.0483 | (-0.0553, -0.0410) | 1.0000 | -0.0483 | (-0.0559, -0.0402) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 0.1023 | (0.0440, 0.1588) | 0.0020 | 0.1023 | (0.0111, 0.1852) | 0.0147 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 0.0681 | (0.0316, 0.1021) | 0.0000 | 0.0681 | (0.0194, 0.1142) | 0.0030 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 0.1643 | (0.1532, 0.1753) | 0.0000 | 0.1643 | (0.1517, 0.1760) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 0.2341 | (0.2195, 0.2494) | 0.0000 | 0.2341 | (0.2156, 0.2506) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 0.1470 | (0.1263, 0.1683) | 0.0000 | 0.1470 | (0.1059, 0.1779) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | 0.0069 | (-0.0080, 0.0212) | 0.1863 | 0.0069 | (-0.0046, 0.0194) | 0.1320 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 0.3053 | (0.2858, 0.3249) | 0.0000 | 0.3053 | (0.2801, 0.3274) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 0.0682 | (0.0618, 0.0747) | 0.0000 | 0.0682 | (0.0597, 0.0775) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 0.1816 | (0.1567, 0.2071) | 0.0000 | 0.1816 | (0.1308, 0.2190) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 0.0086 | (-0.0045, 0.0220) | 0.0973 | 0.0086 | (-0.0062, 0.0250) | 0.1670 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | -0.0472 | (-0.0550, -0.0397) | 1.0000 | -0.0472 | (-0.0545, -0.0390) | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 0.1093 | (0.0458, 0.1731) | 0.0007 | 0.1093 | (0.0574, 0.1669) | 0.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 0.0413 | (0.0097, 0.0753) | 0.0077 | 0.0413 | (-0.0024, 0.0729) | 0.0443 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 0.1651 | (0.1540, 0.1760) | 0.0000 | 0.1651 | (0.1500, 0.1771) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 72 | 25 | 47 | 0.6632 | 0.7423 |
| proposed_vs_candidate_no_context | persona_consistency | 22 | 25 | 97 | 0.4896 | 0.4681 |
| proposed_vs_candidate_no_context | naturalness | 46 | 50 | 48 | 0.4861 | 0.4792 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 59 | 7 | 78 | 0.6806 | 0.8939 |
| proposed_vs_candidate_no_context | context_overlap | 72 | 25 | 47 | 0.6632 | 0.7423 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 12 | 20 | 112 | 0.4722 | 0.3750 |
| proposed_vs_candidate_no_context | persona_style | 13 | 7 | 124 | 0.5208 | 0.6500 |
| proposed_vs_candidate_no_context | distinct1 | 39 | 47 | 58 | 0.4722 | 0.4535 |
| proposed_vs_candidate_no_context | length_score | 45 | 49 | 50 | 0.4861 | 0.4787 |
| proposed_vs_candidate_no_context | sentence_score | 20 | 32 | 92 | 0.4583 | 0.3846 |
| proposed_vs_candidate_no_context | overall_quality | 66 | 32 | 46 | 0.6181 | 0.6735 |
| proposed_vs_baseline_no_context | context_relevance | 88 | 55 | 1 | 0.6146 | 0.6154 |
| proposed_vs_baseline_no_context | persona_consistency | 28 | 34 | 82 | 0.4792 | 0.4516 |
| proposed_vs_baseline_no_context | naturalness | 27 | 117 | 0 | 0.1875 | 0.1875 |
| proposed_vs_baseline_no_context | context_keyword_coverage | 55 | 20 | 69 | 0.6215 | 0.7333 |
| proposed_vs_baseline_no_context | context_overlap | 89 | 54 | 1 | 0.6215 | 0.6224 |
| proposed_vs_baseline_no_context | persona_keyword_coverage | 22 | 27 | 95 | 0.4826 | 0.4490 |
| proposed_vs_baseline_no_context | persona_style | 8 | 14 | 122 | 0.4792 | 0.3636 |
| proposed_vs_baseline_no_context | distinct1 | 27 | 103 | 14 | 0.2361 | 0.2077 |
| proposed_vs_baseline_no_context | length_score | 28 | 112 | 4 | 0.2083 | 0.2000 |
| proposed_vs_baseline_no_context | sentence_score | 19 | 50 | 75 | 0.3924 | 0.2754 |
| proposed_vs_baseline_no_context | overall_quality | 60 | 84 | 0 | 0.4167 | 0.4167 |
| proposed_vs_baseline_no_context_phi3_latest | context_relevance | 87 | 54 | 3 | 0.6146 | 0.6170 |
| proposed_vs_baseline_no_context_phi3_latest | persona_consistency | 25 | 37 | 82 | 0.4583 | 0.4032 |
| proposed_vs_baseline_no_context_phi3_latest | naturalness | 37 | 107 | 0 | 0.2569 | 0.2569 |
| proposed_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 57 | 20 | 67 | 0.6285 | 0.7403 |
| proposed_vs_baseline_no_context_phi3_latest | context_overlap | 89 | 52 | 3 | 0.6285 | 0.6312 |
| proposed_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 20 | 26 | 98 | 0.4792 | 0.4348 |
| proposed_vs_baseline_no_context_phi3_latest | persona_style | 10 | 12 | 122 | 0.4931 | 0.4545 |
| proposed_vs_baseline_no_context_phi3_latest | distinct1 | 29 | 103 | 12 | 0.2431 | 0.2197 |
| proposed_vs_baseline_no_context_phi3_latest | length_score | 36 | 106 | 2 | 0.2569 | 0.2535 |
| proposed_vs_baseline_no_context_phi3_latest | sentence_score | 19 | 61 | 64 | 0.3542 | 0.2375 |
| proposed_vs_baseline_no_context_phi3_latest | overall_quality | 65 | 79 | 0 | 0.4514 | 0.4514 |
| controlled_vs_proposed_raw | context_relevance | 128 | 16 | 0 | 0.8889 | 0.8889 |
| controlled_vs_proposed_raw | persona_consistency | 117 | 9 | 18 | 0.8750 | 0.9286 |
| controlled_vs_proposed_raw | naturalness | 110 | 34 | 0 | 0.7639 | 0.7639 |
| controlled_vs_proposed_raw | context_keyword_coverage | 126 | 14 | 4 | 0.8889 | 0.9000 |
| controlled_vs_proposed_raw | context_overlap | 121 | 23 | 0 | 0.8403 | 0.8403 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 117 | 8 | 19 | 0.8785 | 0.9360 |
| controlled_vs_proposed_raw | persona_style | 12 | 4 | 128 | 0.5278 | 0.7500 |
| controlled_vs_proposed_raw | distinct1 | 66 | 72 | 6 | 0.4792 | 0.4783 |
| controlled_vs_proposed_raw | length_score | 115 | 25 | 4 | 0.8125 | 0.8214 |
| controlled_vs_proposed_raw | sentence_score | 72 | 13 | 59 | 0.7049 | 0.8471 |
| controlled_vs_proposed_raw | overall_quality | 133 | 11 | 0 | 0.9236 | 0.9236 |
| controlled_vs_candidate_no_context | context_relevance | 142 | 2 | 0 | 0.9861 | 0.9861 |
| controlled_vs_candidate_no_context | persona_consistency | 110 | 8 | 26 | 0.8542 | 0.9322 |
| controlled_vs_candidate_no_context | naturalness | 105 | 39 | 0 | 0.7292 | 0.7292 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 141 | 1 | 2 | 0.9861 | 0.9930 |
| controlled_vs_candidate_no_context | context_overlap | 142 | 2 | 0 | 0.9861 | 0.9861 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 109 | 7 | 28 | 0.8542 | 0.9397 |
| controlled_vs_candidate_no_context | persona_style | 20 | 3 | 121 | 0.5590 | 0.8696 |
| controlled_vs_candidate_no_context | distinct1 | 63 | 75 | 6 | 0.4583 | 0.4565 |
| controlled_vs_candidate_no_context | length_score | 111 | 32 | 1 | 0.7743 | 0.7762 |
| controlled_vs_candidate_no_context | sentence_score | 63 | 16 | 65 | 0.6632 | 0.7975 |
| controlled_vs_candidate_no_context | overall_quality | 143 | 1 | 0 | 0.9931 | 0.9931 |
| controlled_vs_baseline_no_context | context_relevance | 143 | 1 | 0 | 0.9931 | 0.9931 |
| controlled_vs_baseline_no_context | persona_consistency | 112 | 7 | 25 | 0.8646 | 0.9412 |
| controlled_vs_baseline_no_context | naturalness | 79 | 64 | 1 | 0.5521 | 0.5524 |
| controlled_vs_baseline_no_context | context_keyword_coverage | 142 | 0 | 2 | 0.9931 | 1.0000 |
| controlled_vs_baseline_no_context | context_overlap | 142 | 2 | 0 | 0.9861 | 0.9861 |
| controlled_vs_baseline_no_context | persona_keyword_coverage | 112 | 2 | 30 | 0.8819 | 0.9825 |
| controlled_vs_baseline_no_context | persona_style | 10 | 8 | 126 | 0.5069 | 0.5556 |
| controlled_vs_baseline_no_context | distinct1 | 20 | 120 | 4 | 0.1528 | 0.1429 |
| controlled_vs_baseline_no_context | length_score | 90 | 49 | 5 | 0.6424 | 0.6475 |
| controlled_vs_baseline_no_context | sentence_score | 44 | 16 | 84 | 0.5972 | 0.7333 |
| controlled_vs_baseline_no_context | overall_quality | 144 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_relevance | 142 | 2 | 0 | 0.9861 | 0.9861 |
| controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 118 | 7 | 19 | 0.8854 | 0.9440 |
| controlled_vs_baseline_no_context_phi3_latest | naturalness | 79 | 65 | 0 | 0.5486 | 0.5486 |
| controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 139 | 0 | 5 | 0.9826 | 1.0000 |
| controlled_vs_baseline_no_context_phi3_latest | context_overlap | 142 | 2 | 0 | 0.9861 | 0.9861 |
| controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 115 | 6 | 23 | 0.8785 | 0.9504 |
| controlled_vs_baseline_no_context_phi3_latest | persona_style | 13 | 4 | 127 | 0.5312 | 0.7647 |
| controlled_vs_baseline_no_context_phi3_latest | distinct1 | 25 | 116 | 3 | 0.1840 | 0.1773 |
| controlled_vs_baseline_no_context_phi3_latest | length_score | 95 | 47 | 2 | 0.6667 | 0.6690 |
| controlled_vs_baseline_no_context_phi3_latest | sentence_score | 34 | 17 | 93 | 0.5590 | 0.6667 |
| controlled_vs_baseline_no_context_phi3_latest | overall_quality | 142 | 2 | 0 | 0.9861 | 0.9861 |
| proposed_contextual_controlled_vs_baseline_no_context | context_relevance | 143 | 1 | 0 | 0.9931 | 0.9931 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_consistency | 112 | 7 | 25 | 0.8646 | 0.9412 |
| proposed_contextual_controlled_vs_baseline_no_context | naturalness | 79 | 64 | 1 | 0.5521 | 0.5524 |
| proposed_contextual_controlled_vs_baseline_no_context | context_keyword_coverage | 142 | 0 | 2 | 0.9931 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context | context_overlap | 142 | 2 | 0 | 0.9861 | 0.9861 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_keyword_coverage | 112 | 2 | 30 | 0.8819 | 0.9825 |
| proposed_contextual_controlled_vs_baseline_no_context | persona_style | 10 | 8 | 126 | 0.5069 | 0.5556 |
| proposed_contextual_controlled_vs_baseline_no_context | distinct1 | 20 | 120 | 4 | 0.1528 | 0.1429 |
| proposed_contextual_controlled_vs_baseline_no_context | length_score | 90 | 49 | 5 | 0.6424 | 0.6475 |
| proposed_contextual_controlled_vs_baseline_no_context | sentence_score | 44 | 16 | 84 | 0.5972 | 0.7333 |
| proposed_contextual_controlled_vs_baseline_no_context | overall_quality | 144 | 0 | 0 | 1.0000 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_relevance | 142 | 2 | 0 | 0.9861 | 0.9861 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_consistency | 118 | 7 | 19 | 0.8854 | 0.9440 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | naturalness | 79 | 65 | 0 | 0.5486 | 0.5486 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_keyword_coverage | 139 | 0 | 5 | 0.9826 | 1.0000 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | context_overlap | 142 | 2 | 0 | 0.9861 | 0.9861 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_keyword_coverage | 115 | 6 | 23 | 0.8785 | 0.9504 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | persona_style | 13 | 4 | 127 | 0.5312 | 0.7647 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | distinct1 | 25 | 116 | 3 | 0.1840 | 0.1773 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | length_score | 95 | 47 | 2 | 0.6667 | 0.6690 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | sentence_score | 34 | 17 | 93 | 0.5590 | 0.6667 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | overall_quality | 142 | 2 | 0 | 0.9861 | 0.9861 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.0139 | 0.4375 | 0.1667 |
| proposed_contextual | 0.0000 | 0.0000 | 0.4583 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.4167 | 0.0000 | 0.0000 |
| baseline_no_context | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline_no_context_phi3_latest | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `8`
- Unique template signatures: `143`
- Template signature ratio: `0.9931`
- Effective sample size by source clustering: `8.00`
- Effective sample size by template-signature clustering: `142.03`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: unavailable (No BERTScore values found in merged scores.).

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
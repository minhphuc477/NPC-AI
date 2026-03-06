# Proposal Alignment Evaluation Report

- Run ID: `20260306T160804Z`
- Generated: `2026-03-06T16:12:47.406821+00:00`
- Scenarios: `artifacts\proposal_control_tuning\20260306T160804Z\scenarios.jsonl`
- Scenario count: `24`

## Evaluation Arms
- `proposed_contextual_controlled`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual_controlled_alt`: model `elara-npc:latest`, dynamic_context=on, response_control=on
- `proposed_contextual`: model `elara-npc:latest`, dynamic_context=on, response_control=off
- `candidate_no_context`: model `elara-npc:latest`, dynamic_context=off, response_control=off

## Metric Summary (mean, 95% CI)
| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.2694 (0.2401, 0.3013) | 0.2846 (0.2399, 0.3314) | 0.8941 (0.8670, 0.9178) | 0.4003 (0.3779, 0.4220) | n/a |
| proposed_contextual_controlled_alt | 0.2893 (0.2529, 0.3307) | 0.3103 (0.2575, 0.3712) | 0.8841 (0.8509, 0.9177) | 0.4167 (0.3910, 0.4421) | n/a |
| proposed_contextual | 0.1081 (0.0500, 0.1762) | 0.1963 (0.1309, 0.2700) | 0.8131 (0.7846, 0.8458) | 0.2764 (0.2296, 0.3296) | n/a |
| candidate_no_context | 0.0402 (0.0212, 0.0617) | 0.2251 (0.1745, 0.2846) | 0.8358 (0.8059, 0.8652) | 0.2595 (0.2339, 0.2864) | n/a |

## Deltas vs Baselines
| Comparison | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0679 | 1.6877 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0288 | -0.1278 |
| proposed_vs_candidate_no_context | naturalness | -0.0226 | -0.0271 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0848 | 2.0919 |
| proposed_vs_candidate_no_context | context_overlap | 0.0283 | 0.7186 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0187 | -0.1582 |
| proposed_vs_candidate_no_context | persona_style | -0.0692 | -0.1058 |
| proposed_vs_candidate_no_context | distinct1 | -0.0043 | -0.0046 |
| proposed_vs_candidate_no_context | length_score | -0.0764 | -0.2068 |
| proposed_vs_candidate_no_context | sentence_score | -0.0583 | -0.0695 |
| proposed_vs_candidate_no_context | overall_quality | 0.0169 | 0.0653 |
| controlled_vs_proposed_raw | context_relevance | 0.1613 | 1.4931 |
| controlled_vs_proposed_raw | persona_consistency | 0.0882 | 0.4494 |
| controlled_vs_proposed_raw | naturalness | 0.0809 | 0.0995 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2176 | 1.7360 |
| controlled_vs_proposed_raw | context_overlap | 0.0302 | 0.4451 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0964 | 0.9720 |
| controlled_vs_proposed_raw | persona_style | 0.0554 | 0.0948 |
| controlled_vs_proposed_raw | distinct1 | 0.0054 | 0.0058 |
| controlled_vs_proposed_raw | length_score | 0.3208 | 1.0948 |
| controlled_vs_proposed_raw | sentence_score | 0.1458 | 0.1867 |
| controlled_vs_proposed_raw | overall_quality | 0.1239 | 0.4483 |
| controlled_vs_candidate_no_context | context_relevance | 0.2292 | 5.7006 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0595 | 0.2642 |
| controlled_vs_candidate_no_context | naturalness | 0.0583 | 0.0697 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3023 | 7.4595 |
| controlled_vs_candidate_no_context | context_overlap | 0.0585 | 1.4835 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0778 | 0.6599 |
| controlled_vs_candidate_no_context | persona_style | -0.0138 | -0.0210 |
| controlled_vs_candidate_no_context | distinct1 | 0.0011 | 0.0012 |
| controlled_vs_candidate_no_context | length_score | 0.2444 | 0.6617 |
| controlled_vs_candidate_no_context | sentence_score | 0.0875 | 0.1042 |
| controlled_vs_candidate_no_context | overall_quality | 0.1409 | 0.5429 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0199 | 0.0740 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0258 | 0.0905 |
| controlled_alt_vs_controlled_default | naturalness | -0.0100 | -0.0112 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0256 | 0.0746 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0068 | 0.0690 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0238 | 0.1217 |
| controlled_alt_vs_controlled_default | persona_style | 0.0335 | 0.0524 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0108 | -0.0114 |
| controlled_alt_vs_controlled_default | length_score | -0.0431 | -0.0701 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0292 | 0.0315 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0164 | 0.0409 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1813 | 1.6775 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1140 | 0.5806 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0709 | 0.0872 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2431 | 1.9401 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0369 | 0.5448 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1202 | 1.2120 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0890 | 0.1521 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0054 | -0.0057 |
| controlled_alt_vs_proposed_raw | length_score | 0.2778 | 0.9479 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1750 | 0.2240 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1403 | 0.5076 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2491 | 6.1962 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.0852 | 0.3786 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0483 | 0.0578 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3279 | 8.0903 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0653 | 1.6548 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1016 | 0.8620 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0198 | 0.0302 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0097 | -0.0102 |
| controlled_alt_vs_candidate_no_context | length_score | 0.2014 | 0.5451 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1167 | 0.1390 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1572 | 0.6060 |

## Paired Bootstrap Delta Significance
| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 0.0679 | (0.0106, 0.1403) | 0.0060 | 0.0679 | (0.0195, 0.1169) | 0.0017 |
| proposed_vs_candidate_no_context | persona_consistency | -0.0288 | (-0.0801, 0.0174) | 0.8770 | -0.0288 | (-0.0806, 0.0154) | 0.8947 |
| proposed_vs_candidate_no_context | naturalness | -0.0226 | (-0.0667, 0.0235) | 0.8333 | -0.0226 | (-0.0902, 0.0225) | 0.8100 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 0.0848 | (0.0066, 0.1749) | 0.0147 | 0.0848 | (0.0208, 0.1480) | 0.0050 |
| proposed_vs_candidate_no_context | context_overlap | 0.0283 | (0.0141, 0.0456) | 0.0000 | 0.0283 | (0.0110, 0.0449) | 0.0000 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | -0.0187 | (-0.0691, 0.0248) | 0.7687 | -0.0187 | (-0.0683, 0.0254) | 0.8003 |
| proposed_vs_candidate_no_context | persona_style | -0.0692 | (-0.1545, 0.0031) | 0.9693 | -0.0692 | (-0.1811, -0.0059) | 1.0000 |
| proposed_vs_candidate_no_context | distinct1 | -0.0043 | (-0.0249, 0.0162) | 0.6520 | -0.0043 | (-0.0383, 0.0190) | 0.6387 |
| proposed_vs_candidate_no_context | length_score | -0.0764 | (-0.2431, 0.0806) | 0.8180 | -0.0764 | (-0.3035, 0.0798) | 0.8010 |
| proposed_vs_candidate_no_context | sentence_score | -0.0583 | (-0.1458, 0.0292) | 0.9317 | -0.0583 | (-0.1500, 0.0109) | 0.9747 |
| proposed_vs_candidate_no_context | overall_quality | 0.0169 | (-0.0292, 0.0638) | 0.2353 | 0.0169 | (-0.0184, 0.0441) | 0.1717 |
| controlled_vs_proposed_raw | context_relevance | 0.1613 | (0.1038, 0.2115) | 0.0000 | 0.1613 | (0.1116, 0.2120) | 0.0000 |
| controlled_vs_proposed_raw | persona_consistency | 0.0882 | (0.0117, 0.1472) | 0.0133 | 0.0882 | (-0.0109, 0.1554) | 0.0347 |
| controlled_vs_proposed_raw | naturalness | 0.0809 | (0.0321, 0.1285) | 0.0007 | 0.0809 | (0.0171, 0.1446) | 0.0030 |
| controlled_vs_proposed_raw | context_keyword_coverage | 0.2176 | (0.1434, 0.2841) | 0.0000 | 0.2176 | (0.1518, 0.2813) | 0.0000 |
| controlled_vs_proposed_raw | context_overlap | 0.0302 | (0.0103, 0.0488) | 0.0020 | 0.0302 | (0.0131, 0.0511) | 0.0000 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 0.0964 | (0.0115, 0.1694) | 0.0133 | 0.0964 | (-0.0245, 0.1845) | 0.0370 |
| controlled_vs_proposed_raw | persona_style | 0.0554 | (-0.0004, 0.1276) | 0.0277 | 0.0554 | (-0.0025, 0.1558) | 0.1007 |
| controlled_vs_proposed_raw | distinct1 | 0.0054 | (-0.0129, 0.0221) | 0.2670 | 0.0054 | (-0.0099, 0.0286) | 0.2743 |
| controlled_vs_proposed_raw | length_score | 0.3208 | (0.1125, 0.5111) | 0.0003 | 0.3208 | (0.0652, 0.5864) | 0.0057 |
| controlled_vs_proposed_raw | sentence_score | 0.1458 | (0.0437, 0.2333) | 0.0043 | 0.1458 | (0.0500, 0.2625) | 0.0007 |
| controlled_vs_proposed_raw | overall_quality | 0.1239 | (0.0753, 0.1699) | 0.0000 | 0.1239 | (0.0732, 0.1674) | 0.0000 |
| controlled_vs_candidate_no_context | context_relevance | 0.2292 | (0.1926, 0.2654) | 0.0000 | 0.2292 | (0.2046, 0.2513) | 0.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 0.0595 | (-0.0028, 0.1159) | 0.0310 | 0.0595 | (-0.0189, 0.1266) | 0.0620 |
| controlled_vs_candidate_no_context | naturalness | 0.0583 | (0.0119, 0.1058) | 0.0053 | 0.0583 | (0.0011, 0.1149) | 0.0227 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 0.3023 | (0.2514, 0.3519) | 0.0000 | 0.3023 | (0.2719, 0.3323) | 0.0000 |
| controlled_vs_candidate_no_context | context_overlap | 0.0585 | (0.0462, 0.0709) | 0.0000 | 0.0585 | (0.0477, 0.0696) | 0.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 0.0778 | (0.0008, 0.1433) | 0.0237 | 0.0778 | (-0.0218, 0.1595) | 0.0550 |
| controlled_vs_candidate_no_context | persona_style | -0.0138 | (-0.0698, 0.0387) | 0.6727 | -0.0138 | (-0.0285, 0.0000) | 1.0000 |
| controlled_vs_candidate_no_context | distinct1 | 0.0011 | (-0.0172, 0.0187) | 0.4587 | 0.0011 | (-0.0162, 0.0164) | 0.4543 |
| controlled_vs_candidate_no_context | length_score | 0.2444 | (0.0694, 0.4194) | 0.0030 | 0.2444 | (0.0060, 0.4923) | 0.0210 |
| controlled_vs_candidate_no_context | sentence_score | 0.0875 | (0.0146, 0.1604) | 0.0117 | 0.0875 | (0.0194, 0.1540) | 0.0040 |
| controlled_vs_candidate_no_context | overall_quality | 0.1409 | (0.1073, 0.1748) | 0.0000 | 0.1409 | (0.1020, 0.1708) | 0.0000 |
| controlled_alt_vs_controlled_default | context_relevance | 0.0199 | (-0.0105, 0.0511) | 0.1087 | 0.0199 | (-0.0084, 0.0435) | 0.0717 |
| controlled_alt_vs_controlled_default | persona_consistency | 0.0258 | (-0.0133, 0.0786) | 0.1280 | 0.0258 | (-0.0141, 0.0969) | 0.1693 |
| controlled_alt_vs_controlled_default | naturalness | -0.0100 | (-0.0432, 0.0206) | 0.7190 | -0.0100 | (-0.0405, 0.0143) | 0.7877 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 0.0256 | (-0.0164, 0.0669) | 0.1137 | 0.0256 | (-0.0091, 0.0534) | 0.0793 |
| controlled_alt_vs_controlled_default | context_overlap | 0.0068 | (-0.0089, 0.0226) | 0.1963 | 0.0068 | (-0.0076, 0.0214) | 0.2223 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 0.0238 | (-0.0198, 0.0843) | 0.1883 | 0.0238 | (-0.0229, 0.1052) | 0.2563 |
| controlled_alt_vs_controlled_default | persona_style | 0.0335 | (-0.0068, 0.0839) | 0.0703 | 0.0335 | (-0.0086, 0.0738) | 0.0613 |
| controlled_alt_vs_controlled_default | distinct1 | -0.0108 | (-0.0254, 0.0020) | 0.9497 | -0.0108 | (-0.0289, 0.0075) | 0.8580 |
| controlled_alt_vs_controlled_default | length_score | -0.0431 | (-0.1972, 0.1028) | 0.7157 | -0.0431 | (-0.1883, 0.0778) | 0.7627 |
| controlled_alt_vs_controlled_default | sentence_score | 0.0292 | (-0.0292, 0.0875) | 0.2203 | 0.0292 | (0.0000, 0.0565) | 0.0997 |
| controlled_alt_vs_controlled_default | overall_quality | 0.0164 | (-0.0051, 0.0390) | 0.0747 | 0.0164 | (-0.0065, 0.0438) | 0.0827 |
| controlled_alt_vs_proposed_raw | context_relevance | 0.1813 | (0.1288, 0.2289) | 0.0000 | 0.1813 | (0.1451, 0.2130) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_consistency | 0.1140 | (0.0714, 0.1543) | 0.0000 | 0.1140 | (0.0720, 0.1593) | 0.0000 |
| controlled_alt_vs_proposed_raw | naturalness | 0.0709 | (0.0211, 0.1224) | 0.0023 | 0.0709 | (0.0013, 0.1475) | 0.0240 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 0.2431 | (0.1744, 0.3088) | 0.0000 | 0.2431 | (0.1942, 0.2840) | 0.0000 |
| controlled_alt_vs_proposed_raw | context_overlap | 0.0369 | (0.0230, 0.0511) | 0.0000 | 0.0369 | (0.0262, 0.0487) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 0.1202 | (0.0748, 0.1639) | 0.0000 | 0.1202 | (0.0685, 0.1724) | 0.0000 |
| controlled_alt_vs_proposed_raw | persona_style | 0.0890 | (0.0208, 0.1698) | 0.0023 | 0.0890 | (0.0050, 0.2086) | 0.0133 |
| controlled_alt_vs_proposed_raw | distinct1 | -0.0054 | (-0.0268, 0.0146) | 0.7023 | -0.0054 | (-0.0320, 0.0251) | 0.6220 |
| controlled_alt_vs_proposed_raw | length_score | 0.2778 | (0.0722, 0.4681) | 0.0057 | 0.2778 | (0.0121, 0.5650) | 0.0193 |
| controlled_alt_vs_proposed_raw | sentence_score | 0.1750 | (0.0875, 0.2625) | 0.0000 | 0.1750 | (0.0875, 0.2763) | 0.0000 |
| controlled_alt_vs_proposed_raw | overall_quality | 0.1403 | (0.1039, 0.1734) | 0.0000 | 0.1403 | (0.1143, 0.1651) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_relevance | 0.2491 | (0.2098, 0.2929) | 0.0000 | 0.2491 | (0.2174, 0.2739) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 0.0852 | (0.0418, 0.1259) | 0.0003 | 0.0852 | (0.0415, 0.1355) | 0.0000 |
| controlled_alt_vs_candidate_no_context | naturalness | 0.0483 | (0.0001, 0.0961) | 0.0243 | 0.0483 | (-0.0146, 0.1079) | 0.0627 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 0.3279 | (0.2781, 0.3856) | 0.0000 | 0.3279 | (0.2882, 0.3566) | 0.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 0.0653 | (0.0514, 0.0810) | 0.0000 | 0.0653 | (0.0476, 0.0817) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 0.1016 | (0.0550, 0.1494) | 0.0000 | 0.1016 | (0.0461, 0.1587) | 0.0000 |
| controlled_alt_vs_candidate_no_context | persona_style | 0.0198 | (-0.0168, 0.0578) | 0.1630 | 0.0198 | (-0.0122, 0.0519) | 0.1230 |
| controlled_alt_vs_candidate_no_context | distinct1 | -0.0097 | (-0.0318, 0.0117) | 0.8150 | -0.0097 | (-0.0371, 0.0186) | 0.7613 |
| controlled_alt_vs_candidate_no_context | length_score | 0.2014 | (0.0083, 0.3931) | 0.0223 | 0.2014 | (-0.0400, 0.4152) | 0.0563 |
| controlled_alt_vs_candidate_no_context | sentence_score | 0.1167 | (0.0583, 0.1896) | 0.0000 | 0.1167 | (0.0350, 0.1820) | 0.0047 |
| controlled_alt_vs_candidate_no_context | overall_quality | 0.1572 | (0.1288, 0.1838) | 0.0000 | 0.1572 | (0.1355, 0.1780) | 0.0000 |

## Pairwise Win Rates
| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |
|---|---|---:|---:|---:|---:|---:|
| proposed_vs_candidate_no_context | context_relevance | 13 | 5 | 6 | 0.6667 | 0.7222 |
| proposed_vs_candidate_no_context | persona_consistency | 5 | 8 | 11 | 0.4375 | 0.3846 |
| proposed_vs_candidate_no_context | naturalness | 9 | 9 | 6 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | context_keyword_coverage | 8 | 4 | 12 | 0.5833 | 0.6667 |
| proposed_vs_candidate_no_context | context_overlap | 15 | 3 | 6 | 0.7500 | 0.8333 |
| proposed_vs_candidate_no_context | persona_keyword_coverage | 4 | 5 | 15 | 0.4792 | 0.4444 |
| proposed_vs_candidate_no_context | persona_style | 2 | 7 | 15 | 0.3958 | 0.2222 |
| proposed_vs_candidate_no_context | distinct1 | 8 | 9 | 7 | 0.4792 | 0.4706 |
| proposed_vs_candidate_no_context | length_score | 9 | 9 | 6 | 0.5000 | 0.5000 |
| proposed_vs_candidate_no_context | sentence_score | 3 | 7 | 14 | 0.4167 | 0.3000 |
| proposed_vs_candidate_no_context | overall_quality | 10 | 8 | 6 | 0.5417 | 0.5556 |
| controlled_vs_proposed_raw | context_relevance | 21 | 3 | 0 | 0.8750 | 0.8750 |
| controlled_vs_proposed_raw | persona_consistency | 16 | 4 | 4 | 0.7500 | 0.8000 |
| controlled_vs_proposed_raw | naturalness | 19 | 5 | 0 | 0.7917 | 0.7917 |
| controlled_vs_proposed_raw | context_keyword_coverage | 20 | 2 | 2 | 0.8750 | 0.9091 |
| controlled_vs_proposed_raw | context_overlap | 20 | 4 | 0 | 0.8333 | 0.8333 |
| controlled_vs_proposed_raw | persona_keyword_coverage | 16 | 4 | 4 | 0.7500 | 0.8000 |
| controlled_vs_proposed_raw | persona_style | 6 | 1 | 17 | 0.6042 | 0.8571 |
| controlled_vs_proposed_raw | distinct1 | 13 | 9 | 2 | 0.5833 | 0.5909 |
| controlled_vs_proposed_raw | length_score | 18 | 5 | 1 | 0.7708 | 0.7826 |
| controlled_vs_proposed_raw | sentence_score | 13 | 3 | 8 | 0.7083 | 0.8125 |
| controlled_vs_proposed_raw | overall_quality | 20 | 4 | 0 | 0.8333 | 0.8333 |
| controlled_vs_candidate_no_context | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_consistency | 13 | 5 | 6 | 0.6667 | 0.7222 |
| controlled_vs_candidate_no_context | naturalness | 15 | 9 | 0 | 0.6250 | 0.6250 |
| controlled_vs_candidate_no_context | context_keyword_coverage | 23 | 0 | 1 | 0.9792 | 1.0000 |
| controlled_vs_candidate_no_context | context_overlap | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_vs_candidate_no_context | persona_keyword_coverage | 12 | 3 | 9 | 0.6875 | 0.8000 |
| controlled_vs_candidate_no_context | persona_style | 3 | 5 | 16 | 0.4583 | 0.3750 |
| controlled_vs_candidate_no_context | distinct1 | 12 | 11 | 1 | 0.5208 | 0.5217 |
| controlled_vs_candidate_no_context | length_score | 16 | 7 | 1 | 0.6875 | 0.6957 |
| controlled_vs_candidate_no_context | sentence_score | 7 | 1 | 16 | 0.6250 | 0.8750 |
| controlled_vs_candidate_no_context | overall_quality | 23 | 1 | 0 | 0.9583 | 0.9583 |
| controlled_alt_vs_controlled_default | context_relevance | 8 | 7 | 9 | 0.5208 | 0.5333 |
| controlled_alt_vs_controlled_default | persona_consistency | 6 | 2 | 16 | 0.5833 | 0.7500 |
| controlled_alt_vs_controlled_default | naturalness | 8 | 8 | 8 | 0.5000 | 0.5000 |
| controlled_alt_vs_controlled_default | context_keyword_coverage | 7 | 3 | 14 | 0.5833 | 0.7000 |
| controlled_alt_vs_controlled_default | context_overlap | 9 | 6 | 9 | 0.5625 | 0.6000 |
| controlled_alt_vs_controlled_default | persona_keyword_coverage | 3 | 2 | 19 | 0.5208 | 0.6000 |
| controlled_alt_vs_controlled_default | persona_style | 4 | 1 | 19 | 0.5625 | 0.8000 |
| controlled_alt_vs_controlled_default | distinct1 | 5 | 10 | 9 | 0.3958 | 0.3333 |
| controlled_alt_vs_controlled_default | length_score | 6 | 8 | 10 | 0.4583 | 0.4286 |
| controlled_alt_vs_controlled_default | sentence_score | 3 | 1 | 20 | 0.5417 | 0.7500 |
| controlled_alt_vs_controlled_default | overall_quality | 10 | 6 | 8 | 0.5833 | 0.6250 |
| controlled_alt_vs_proposed_raw | context_relevance | 20 | 4 | 0 | 0.8333 | 0.8333 |
| controlled_alt_vs_proposed_raw | persona_consistency | 17 | 1 | 6 | 0.8333 | 0.9444 |
| controlled_alt_vs_proposed_raw | naturalness | 17 | 7 | 0 | 0.7083 | 0.7083 |
| controlled_alt_vs_proposed_raw | context_keyword_coverage | 20 | 3 | 1 | 0.8542 | 0.8696 |
| controlled_alt_vs_proposed_raw | context_overlap | 20 | 4 | 0 | 0.8333 | 0.8333 |
| controlled_alt_vs_proposed_raw | persona_keyword_coverage | 16 | 1 | 7 | 0.8125 | 0.9412 |
| controlled_alt_vs_proposed_raw | persona_style | 8 | 1 | 15 | 0.6458 | 0.8889 |
| controlled_alt_vs_proposed_raw | distinct1 | 12 | 10 | 2 | 0.5417 | 0.5455 |
| controlled_alt_vs_proposed_raw | length_score | 17 | 6 | 1 | 0.7292 | 0.7391 |
| controlled_alt_vs_proposed_raw | sentence_score | 13 | 1 | 10 | 0.7500 | 0.9286 |
| controlled_alt_vs_proposed_raw | overall_quality | 22 | 2 | 0 | 0.9167 | 0.9167 |
| controlled_alt_vs_candidate_no_context | context_relevance | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_consistency | 15 | 1 | 8 | 0.7917 | 0.9375 |
| controlled_alt_vs_candidate_no_context | naturalness | 14 | 10 | 0 | 0.5833 | 0.5833 |
| controlled_alt_vs_candidate_no_context | context_keyword_coverage | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | context_overlap | 24 | 0 | 0 | 1.0000 | 1.0000 |
| controlled_alt_vs_candidate_no_context | persona_keyword_coverage | 14 | 1 | 9 | 0.7708 | 0.9333 |
| controlled_alt_vs_candidate_no_context | persona_style | 4 | 2 | 18 | 0.5417 | 0.6667 |
| controlled_alt_vs_candidate_no_context | distinct1 | 11 | 13 | 0 | 0.4583 | 0.4583 |
| controlled_alt_vs_candidate_no_context | length_score | 16 | 8 | 0 | 0.6667 | 0.6667 |
| controlled_alt_vs_candidate_no_context | sentence_score | 8 | 0 | 16 | 0.6667 | 1.0000 |
| controlled_alt_vs_candidate_no_context | overall_quality | 24 | 0 | 0 | 1.0000 | 1.0000 |

## Scenario Slice Coverage
- Slice keys: `persona_archetype, conflict_type, location_type, behavior_state`
- Detailed slice metrics are published in `slice_summary.json`.

## Operational Metrics
| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |
|---|---:|---:|---:|---:|---:|
| proposed_contextual_controlled | 0.0000 | 0.0000 | 0.1250 | 0.5417 | 0.1250 |
| proposed_contextual_controlled_alt | 0.0000 | 0.0000 | 0.1250 | 0.5833 | 0.1250 |
| proposed_contextual | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 |
| candidate_no_context | 0.0000 | 0.0000 | 0.3333 | 0.0000 | 0.0000 |

## Scenario Dependence Diagnostics
- Unique source scenarios: `8`
- Unique template signatures: `20`
- Template signature ratio: `0.8333`
- Effective sample size by source clustering: `6.86`
- Effective sample size by template-signature clustering: `18.00`
- Detailed diagnostics are published in `scenario_dependence.json`.

- BERTScore status: unavailable (disabled_by_flag).

This report directly covers proposal RO5 metrics: context relevance, persona consistency, naturalness, and baseline deltas.
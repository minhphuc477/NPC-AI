# Proposal Quality Gate Report

- Generated UTC: `2026-03-12T15:03:14.561054+00:00`
- Overall pass: `False`
- Proposal run: `artifacts\proposal\20260312T071603Z`
- Publication run: `artifacts\publication\20260312T104035Z`

## Checks
| Section | Check | Result | Details |
|---|---|---|---|
| proposal | file:run_config.json | PASS | artifacts\proposal\20260312T071603Z\run_config.json |
| proposal | file:comparison_plan.json | PASS | artifacts\proposal\20260312T071603Z\comparison_plan.json |
| proposal | file:summary.json | PASS | artifacts\proposal\20260312T071603Z\summary.json |
| proposal | file:paired_delta_significance.json | PASS | artifacts\proposal\20260312T071603Z\paired_delta_significance.json |
| proposal | file:win_rates.json | PASS | artifacts\proposal\20260312T071603Z\win_rates.json |
| proposal | file:slice_summary.json | PASS | artifacts\proposal\20260312T071603Z\slice_summary.json |
| proposal | file:report.md | PASS | artifacts\proposal\20260312T071603Z\report.md |
| proposal | file:scenarios.jsonl | PASS | artifacts\proposal\20260312T071603Z\scenarios.jsonl |
| proposal | file:metadata/hardware.json | PASS | artifacts\proposal\20260312T071603Z\metadata\hardware.json |
| proposal | file:metadata/models.json | PASS | artifacts\proposal\20260312T071603Z\metadata\models.json |
| proposal | scenario_coverage | PASS | rows=144, min_required=100 |
| proposal | bertscore_repro_config | PASS | model_type='roberta-large', batch_size=16 |
| proposal | controlled_vs_raw:context_relevance | PASS | comparison=controlled_vs_proposed_raw, mean_delta=0.034744, p_delta_le_0=0.001333, alpha=0.05 |
| proposal | controlled_vs_raw:persona_consistency | PASS | comparison=controlled_vs_proposed_raw, mean_delta=0.087424, p_delta_le_0=0.000000, alpha=0.05 |
| proposal | controlled_vs_raw:naturalness | FAIL | comparison=controlled_vs_proposed_raw, mean_delta=0.002255, p_delta_le_0=0.327667, alpha=0.05 |
| proposal | controlled_vs_raw:overall_quality | PASS | comparison=controlled_vs_proposed_raw, mean_delta=0.049364, p_delta_le_0=0.000000, alpha=0.05 |
| proposal | external_wins:baseline_no_context | PASS | comparison=controlled_vs_baseline_no_context, significant_positive=11, min_required=10 |
| proposal | external_overall_quality:baseline_no_context | PASS | comparison=controlled_vs_baseline_no_context, overall_quality_delta=0.086166, p_delta_le_0=0.000000, alpha=0.05 |
| proposal | external_wins:baseline_no_context_phi3_latest | PASS | comparison=controlled_vs_baseline_no_context_phi3_latest, significant_positive=12, min_required=10 |
| proposal | external_overall_quality:baseline_no_context_phi3_latest | PASS | comparison=controlled_vs_baseline_no_context_phi3_latest, overall_quality_delta=0.089401, p_delta_le_0=0.000000, alpha=0.05 |
| proposal | human_eval_present | PASS | artifacts\proposal\20260312T071603Z\human_eval_summary.json |
| proposal | human_eval_row_count | PASS | rows=324, min_required=300 |
| proposal | human_eval_agreement | PASS | min_pair_count=3, mean_pairwise_kappa=0.542621, min_kappa_required=0.200000 |
| proposal | human_pref_soft_win:baseline_no_context | PASS | comparison=proposed_contextual_controlled_vs_baseline_no_context, soft_win_rate=0.712963, min_required=0.500000 |
| proposal | human_pref_soft_win:baseline_no_context_phi3_latest | PASS | comparison=proposed_contextual_controlled_vs_baseline_no_context_phi3_latest, soft_win_rate=0.532407, min_required=0.500000 |
| publication | file:run_config.json | PASS | artifacts\publication\20260312T104035Z\run_config.json |
| publication | file:metadata/hardware.json | PASS | artifacts\publication\20260312T104035Z\metadata\hardware.json |
| publication | file:metadata/models.json | PASS | artifacts\publication\20260312T104035Z\metadata\models.json |
| publication | file:serving/summary.json | PASS | artifacts\publication\20260312T104035Z\serving\summary.json |
| publication | file:serving/delta_vs_baseline.json | PASS | artifacts\publication\20260312T104035Z\serving\delta_vs_baseline.json |
| publication | file:retrieval/metrics.json | PASS | artifacts\publication\20260312T104035Z\retrieval\metrics.json |
| publication | file:retrieval/ablation_deltas_vs_bm25.json | PASS | artifacts\publication\20260312T104035Z\retrieval\ablation_deltas_vs_bm25.json |
| publication | file:report.md | PASS | artifacts\publication\20260312T104035Z\report.md |
| publication | serving_models_declared | PASS | candidate='elara-npc:latest', baseline='phi3:mini' |
| publication | identical_inputs_declared | PASS | prompts='data\serving_prompts_wide_v2.jsonl', references='data\serving_references_wide_v2.jsonl', retrieval_gold='data\retrieval_gold_wide_v2.jsonl', retrieval_corpus='data\retrieval_corpus_wide_v2.jsonl' |
| publication | serving_confidence_intervals | PASS | ttft_ms/total_time_ms/tokens_per_s each include mean + 95% CI for candidate and baseline |
| publication | serving_delta_present | PASS | delta_metrics=['ttft_ms', 'total_time_ms', 'tokens_per_s'] |
| publication | retrieval_metrics_standardized | PASS | bm25_keys=['hit@5', 'mrr', 'ndcg@5', 'query_count', 'by_query_type'] |
| publication | retrieval_ablation_deltas_present | PASS | ablation_methods=['keyword_overlap', 'random'] |
| publication | serving_prompt_parity | PASS | request_files=2 |
| publication | security_benchmark_present | PASS | artifacts\publication\20260312T104035Z\retrieval\security_guard_benchmark_spoofed.json |
| publication | security_asr_reduction | PASS | baseline_asr=1.000000, guarded_asr=0.000000, relative_reduction=1.000000, min_reduction=0.800000, max_guarded_asr=0.050000, alpha=0.05 |
# Proposal Quality Gate Report

- Generated UTC: `2026-02-25T05:26:14.007307+00:00`
- Overall pass: `True`
- Proposal run: `artifacts\proposal\20260224T175344Z`
- Publication run: `artifacts\publication\20260224T151628Z`

## Checks
| Section | Check | Result | Details |
|---|---|---|---|
| proposal | file:run_config.json | PASS | artifacts\proposal\20260224T175344Z\run_config.json |
| proposal | file:comparison_plan.json | PASS | artifacts\proposal\20260224T175344Z\comparison_plan.json |
| proposal | file:summary.json | PASS | artifacts\proposal\20260224T175344Z\summary.json |
| proposal | file:paired_delta_significance.json | PASS | artifacts\proposal\20260224T175344Z\paired_delta_significance.json |
| proposal | file:win_rates.json | PASS | artifacts\proposal\20260224T175344Z\win_rates.json |
| proposal | file:slice_summary.json | PASS | artifacts\proposal\20260224T175344Z\slice_summary.json |
| proposal | file:report.md | PASS | artifacts\proposal\20260224T175344Z\report.md |
| proposal | file:scenarios.jsonl | PASS | artifacts\proposal\20260224T175344Z\scenarios.jsonl |
| proposal | file:metadata/hardware.json | PASS | artifacts\proposal\20260224T175344Z\metadata\hardware.json |
| proposal | file:metadata/models.json | PASS | artifacts\proposal\20260224T175344Z\metadata\models.json |
| proposal | scenario_coverage | PASS | rows=112, min_required=100 |
| proposal | bertscore_repro_config | PASS | model_type='roberta-large', batch_size=16 |
| proposal | controlled_vs_raw:context_relevance | PASS | comparison=controlled_vs_proposed_raw, mean_delta=0.204863, p_delta_le_0=0.000000, alpha=0.05 |
| proposal | controlled_vs_raw:persona_consistency | PASS | comparison=controlled_vs_proposed_raw, mean_delta=0.078738, p_delta_le_0=0.000333, alpha=0.05 |
| proposal | controlled_vs_raw:naturalness | PASS | comparison=controlled_vs_proposed_raw, mean_delta=0.088531, p_delta_le_0=0.000000, alpha=0.05 |
| proposal | controlled_vs_raw:overall_quality | PASS | comparison=controlled_vs_proposed_raw, mean_delta=0.128390, p_delta_le_0=0.000000, alpha=0.05 |
| proposal | external_wins:baseline_no_context | PASS | comparison=controlled_vs_baseline_no_context, significant_positive=10, min_required=10 |
| proposal | external_overall_quality:baseline_no_context | PASS | comparison=controlled_vs_baseline_no_context, overall_quality_delta=0.125638, p_delta_le_0=0.000000, alpha=0.05 |
| proposal | external_wins:baseline_no_context_phi3_latest | PASS | comparison=controlled_vs_baseline_no_context_phi3_latest, significant_positive=11, min_required=10 |
| proposal | external_overall_quality:baseline_no_context_phi3_latest | PASS | comparison=controlled_vs_baseline_no_context_phi3_latest, overall_quality_delta=0.120949, p_delta_le_0=0.000000, alpha=0.05 |
| proposal | human_eval_present | PASS | artifacts\proposal\20260224T175344Z\human_eval_summary.json |
| proposal | human_eval_row_count | PASS | rows=324, min_required=300 |
| proposal | human_eval_agreement | PASS | min_pair_count=3, mean_pairwise_kappa=0.425820, min_kappa_required=0.200000 |
| proposal | human_pref_soft_win:baseline_no_context | PASS | comparison=proposed_contextual_controlled_vs_baseline_no_context, soft_win_rate=0.736111, min_required=0.550000 |
| proposal | human_pref_soft_win:baseline_no_context_phi3_latest | PASS | comparison=proposed_contextual_controlled_vs_baseline_no_context_phi3_latest, soft_win_rate=0.671296, min_required=0.550000 |
| publication | file:run_config.json | PASS | artifacts\publication\20260224T151628Z\run_config.json |
| publication | file:metadata/hardware.json | PASS | artifacts\publication\20260224T151628Z\metadata\hardware.json |
| publication | file:metadata/models.json | PASS | artifacts\publication\20260224T151628Z\metadata\models.json |
| publication | file:serving/summary.json | PASS | artifacts\publication\20260224T151628Z\serving\summary.json |
| publication | file:serving/delta_vs_baseline.json | PASS | artifacts\publication\20260224T151628Z\serving\delta_vs_baseline.json |
| publication | file:retrieval/metrics.json | PASS | artifacts\publication\20260224T151628Z\retrieval\metrics.json |
| publication | file:retrieval/ablation_deltas_vs_bm25.json | PASS | artifacts\publication\20260224T151628Z\retrieval\ablation_deltas_vs_bm25.json |
| publication | file:report.md | PASS | artifacts\publication\20260224T151628Z\report.md |
| publication | serving_models_declared | PASS | candidate='elara-npc:latest', baseline='phi3:mini' |
| publication | identical_inputs_declared | PASS | prompts='data\serving_prompts.jsonl', references='data\serving_references.jsonl', retrieval_gold='data\retrieval_gold.jsonl', retrieval_corpus='data\retrieval_corpus.jsonl' |
| publication | serving_confidence_intervals | PASS | ttft_ms/total_time_ms/tokens_per_s each include mean + 95% CI for candidate and baseline |
| publication | serving_delta_present | PASS | delta_metrics=['ttft_ms', 'total_time_ms', 'tokens_per_s'] |
| publication | retrieval_metrics_standardized | PASS | bm25_keys=['hit@5', 'mrr', 'ndcg@5', 'query_count'] |
| publication | retrieval_ablation_deltas_present | PASS | ablation_methods=['keyword_overlap', 'random'] |
| publication | serving_prompt_parity | PASS | request_files=2 |
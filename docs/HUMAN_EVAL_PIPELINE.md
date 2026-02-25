# Human Evaluation Pipeline

## Purpose
Run stronger human evaluation in the same proposal artifact pipeline with:
- per-arm metric confidence intervals,
- inter-rater agreement (pairwise Cohen's kappa),
- paired deltas and preference win rates versus baselines.

## 1. Generate/refresh expanded scenario set
```bash
python scripts/generate_proposal_scenarios_large.py --variants-per-base 14 --output data/proposal_eval_scenarios_large.jsonl
```

## 2. Create a rating template
```bash
python scripts/create_human_eval_template.py --scenarios data/proposal_eval_scenarios_large.jsonl --output data/proposal_human_eval_template.csv
```

Or build a blind, multi-rater pack directly from a proposal run:
```bash
python scripts/build_human_eval_pack.py --run-dir artifacts/proposal/<run_id> --annotators "annotator_1,annotator_2,annotator_3" --shared-ratio 0.35
```

Expected columns:
- `scenario_id`
- `arm_id`
- `annotator_id`
- `context_relevance`
- `persona_consistency`
- `naturalness`
- `overall_quality`
- `notes` (optional)

Rating scale:
- default expected range is `1..5` and normalized to `0..1` internally.

## 3A. Attach external human ratings during proposal run
```bash
python scripts/run_proposal_alignment_eval_batched.py ^
  --scenarios data/proposal_eval_scenarios_large.jsonl ^
  --batch-size 28 ^
  --baseline-models "phi3:latest" ^
  --human-eval-file data/proposal_human_eval_ratings.csv ^
  --human-eval-metrics "context_relevance,persona_consistency,naturalness,overall_quality" ^
  --human-eval-scale-max 5
```

## 3B. LLM multi-rater campaign on an existing proposal run
```bash
python scripts/run_llm_multirater_campaign.py ^
  --run-dir artifacts/proposal/<run_id> ^
  --annotators "phi3:mini|balanced|0.00,phi3:mini|balanced|0.05,phi3:mini|balanced|0.10" ^
  --arms "proposed_contextual_controlled,baseline_no_context,baseline_no_context_phi3_latest" ^
  --scenario-limit 36 ^
  --output-csv artifacts/proposal/<run_id>/human_eval_llm_multirater_consistent.csv

python scripts/attach_human_eval_to_run.py ^
  --run-dir artifacts/proposal/<run_id> ^
  --human-eval-file artifacts/proposal/<run_id>/human_eval_llm_multirater_consistent.csv
```

## 4. Output artifacts
Inside `artifacts/proposal/<run_id>/`:
- `win_rates.json`
- `slice_summary.json`
- `comparison_plan.json`
- `human_eval_summary.json`
- `human_eval_report.md`
- `human_eval_llm_multirater_consistent.csv` (if LLM campaign route is used)

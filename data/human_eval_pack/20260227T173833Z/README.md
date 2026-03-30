# Human Eval Pack

## Files
- `<annotator>.csv`: per-rater blind worksheet.
- `blind_mapping.json`: hidden mapping from blind IDs to true arms.
- `assignment_plan.json`: scenario split and expected row counts.
- `ratings_merged_template.csv`: merge target file.

## Workflow
1. Send each annotator only their own CSV.
2. Collect completed files and concatenate rows into `ratings_merged_template.csv`.
3. Run proposal evaluation with:
```bash
python scripts/run_proposal_alignment_eval_batched.py --scenarios artifacts\proposal\20260227T173833Z\scenarios.jsonl --human-eval-file data\human_eval_pack\20260227T173833Z\ratings_merged_template.csv
```
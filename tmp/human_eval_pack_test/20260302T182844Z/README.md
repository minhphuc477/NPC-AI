# Human Eval Pack

## Files
- `<annotator>.csv`: per-rater blind worksheet.
- `blind_mapping.json`: hidden mapping from blind IDs to true arms.
- `assignment_plan.json`: scenario split and expected row counts.
- `ratings_merged_template.csv`: merge target file.

## Study Design Notes
- Ratings per non-shared scenario: `2`.
- `assignment_plan.json` contains overlap matrix and power-based sample-size planning.

## Workflow
1. Send each annotator only their own CSV.
2. Collect completed files and concatenate rows into `ratings_merged_template.csv`.
3. Run proposal evaluation with:
```bash
python scripts/run_proposal_alignment_eval_batched.py --scenarios artifacts\proposal\20260302T182844Z\scenarios.jsonl --human-eval-file tmp\human_eval_pack_test\20260302T182844Z\ratings_merged_template.csv
```
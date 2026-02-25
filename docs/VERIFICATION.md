# Verification Log

## Scope
Record what has been executed and verified in this environment.

## Verified
1. Unified full checkout pipeline executes successfully:
- `scripts/run_kaggle_full_results.py`

2. Full checkout produced manifest:
- `artifacts/final_checkout/20260225T052614Z/manifest.json`

3. Proposal run has attached multi-rater evaluation artifacts:
- `human_eval_llm_multirater_consistent.csv`
- `human_eval_summary.json`
- `human_eval_report.md`

4. Additional analysis artifacts regenerate correctly:
- lexical diversity benchmark
- preference dataset builder
- retrieval hard-negative set builder

5. Quality gate passes (human-eval required):
- `artifacts/proposal/20260224T175344Z/quality_gate_report_final.md`

6. Notebook has full checkout execution cell:
- `notebooks/NPC_AI_Complete_Pipeline.ipynb` includes `scripts/run_kaggle_full_results.py`

## Not Fully Verified
1. Security benchmark requirement on Kaggle unless retrieval security executable is available.
2. Serving superiority claim over lightweight baselines (current evidence still negative on quality frontier).
3. Full paper-protocol replication against external publications.

## Reproduce
```bash
python scripts/run_kaggle_full_results.py --host http://127.0.0.1:11434
```

Resume after interruption:
```bash
python scripts/run_kaggle_full_results.py --host http://127.0.0.1:11434 --proposal-run latest --publication-run latest
```

# Kaggle Guide (Full Results Checkout)

## Scope
Run `notebooks/NPC_AI_Complete_Pipeline.ipynb` on Kaggle and produce a complete artifact bundle:
- proposal run,
- human-eval attachment,
- lexical benchmark,
- publication run,
- serving-efficiency matrix,
- external profile suite,
- quality-gate report,
- preference/hard-negative datasets.

## Required Inputs
1. Repository snapshot as a Kaggle Dataset (recommended).
2. Internet enabled (model/dependency pulls).
3. Local model runtime reachable from notebook process (`ollama` host), with:
- `elara-npc:latest`
- `phi3:mini`
- `phi3:latest`

## Setup
1. Upload/attach this repo in Kaggle (`Add Data`).
2. Upload `notebooks/NPC_AI_Complete_Pipeline.ipynb`.
3. Refresh notebook with the latest orchestration cells before upload:
```bash
python scripts/create_pipeline_notebook.py --clear-outputs
```
4. Use GPU runtime (T4/P100 preferred for stability).

## One-Command Full Run
Use this command in notebook (or the generated "Full Artifact Checkout" cell):
```bash
python scripts/run_kaggle_full_results.py --host http://127.0.0.1:11434
```

Ablation option (disable keyword/random retrieval ablation baselines):
```bash
python scripts/run_kaggle_full_results.py --host http://127.0.0.1:11434 --skip-ablation-baselines
```

Optional security benchmark mode (requires compiled retrieval security executable):
```bash
python scripts/run_kaggle_full_results.py \
  --host http://127.0.0.1:11434 \
  --run-security-benchmark \
  --require-security-benchmark
```

Resume from existing runs (recommended after timeout/interruption):
```bash
python scripts/run_kaggle_full_results.py \
  --host http://127.0.0.1:11434 \
  --proposal-run latest \
  --publication-run latest
```

## Outputs
Primary manifest:
- `artifacts/final_checkout/<timestamp>/manifest.json`

Referenced artifacts include:
- `artifacts/proposal/<run_id>/...`
- `artifacts/publication/<run_id>/...`
- `artifacts/serving_efficiency/<run_id>/...`
- `artifacts/publication_profiles/<run_id>/...`
- `artifacts/proposal/<run_id>/quality_gate_report_final.md`
- `artifacts/proposal/<run_id>/preference_dataset.jsonl`
- `data/retrieval_hard_negatives_wide.jsonl`
- `data/retrieval_reranker_pairs_wide.jsonl`

## Common Failure Modes
1. `Ollama host is unavailable`
- Ensure the runtime can reach `--host` and the server is running.

2. `Missing Ollama models`
- Pull/create required models before full checkout run.

3. Security benchmark requirement fails
- Either compile/provide `bench_retrieval_security` or run without `--require-security-benchmark`.

4. Notebook cannot find scripts
- Attach repo snapshot in `Add Data` and run from repo root.

## Minimal Success Criteria
1. `artifacts/final_checkout/<timestamp>/manifest.json` exists.
2. `quality_gate_report_final.md` shows pass for required checks.
3. Proposal run contains:
- `human_eval_summary.json`
- `lexical_diversity_summary.json`
- `preference_dataset.jsonl`

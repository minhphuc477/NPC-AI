# Proposal Quality Bar

## Scope
Define a strict, reproducible pass/fail gate for proposal-quality claims.

## Quality Gate Script
- `scripts/proposal_quality_gate.py`

It validates both:
- proposal artifacts (`artifacts/proposal/<run_id>`)
- publication artifacts (`artifacts/publication/<run_id>`)

## Default Checks

### Proposal checks
1. Required artifact files exist (`summary.json`, `paired_delta_significance.json`, metadata, report).
2. Scenario coverage meets minimum (`--min-scenarios`, default `100`).
3. Controlled arm significantly improves key metrics over raw contextual arm:
- `context_relevance`
- `persona_consistency`
- `naturalness`
- `overall_quality`
4. Controlled arm has broad significant wins versus each external baseline:
- minimum significantly-positive metrics per baseline (`--min-external-significant-wins`, default `10`)
- `overall_quality` must be significantly positive.
5. BERTScore reproducibility config is captured in run config.
6. Optional: completed human evaluation quality checks (`--require-human-eval`).

### Publication checks
1. Non-mock metadata files exist (`metadata/hardware.json`, `metadata/models.json`).
2. Standard retrieval metrics exist with confidence intervals (`Hit@k`, `MRR`, `nDCG@k`).
3. Ablation deltas are published.
4. Serving comparison outputs are present and include confidence intervals.
5. Prompt parity across compared serving runs is verified.
6. Optional: security benchmark threshold checks (`--require-security-benchmark`).

## Recommended Run Order
1. Run proposal pipeline in batches:
```bash
python scripts/run_proposal_alignment_eval_batched.py \
  --scenarios data/proposal_eval_scenarios_large.jsonl \
  --batch-size 28 \
  --baseline-models "phi3:latest"
```

2. Build blind human-eval pack from the generated proposal run:
```bash
python scripts/build_human_eval_pack.py \
  --run-dir artifacts/proposal/<run_id> \
  --annotators "annotator_1,annotator_2,annotator_3" \
  --shared-ratio 0.35
```

3. Re-run proposal pipeline with merged human ratings:
```bash
python scripts/run_proposal_alignment_eval_batched.py \
  --scenarios data/proposal_eval_scenarios_large.jsonl \
  --human-eval-file data/human_eval_pack/<run_id>/ratings_merged_template.csv \
  --batch-size 28 \
  --baseline-models "phi3:latest"
```

4. Run publication suite:
```bash
python scripts/run_publication_benchmark_suite.py \
  --repeats 1 \
  --max-tokens 64 \
  --temperature 0.2 \
  --run-security-benchmark \
  --run-security-spoofed-benchmark
```

5. Run strict quality gate:
```bash
python scripts/proposal_quality_gate.py \
  --proposal-run latest \
  --publication-run latest \
  --require-human-eval \
  --require-security-benchmark \
  --strict
```

## One-command Gate from Proposal Batched Runner
You can trigger gate execution right after a batched proposal run:
```bash
python scripts/run_proposal_alignment_eval_batched.py \
  --scenarios data/proposal_eval_scenarios_large.jsonl \
  --batch-size 28 \
  --baseline-models "phi3:latest" \
  --quality-gate \
  --quality-gate-publication-run latest \
  --quality-gate-require-human-eval \
  --quality-gate-require-security-benchmark \
  --quality-gate-strict
```

## Output
The gate writes:
- `quality_gate_report.json`
- `quality_gate_report.md`

By default these are written under the proposal run directory.

## Full Kaggle Checkout (Unified)
For one-command proposal/publication checkout with attached LLM multi-rater eval, lexical benchmark,
serving matrix, external profiles, and final gate output:
```bash
python scripts/run_kaggle_full_results.py --host http://127.0.0.1:11434
```

# NPC AI Benchmark Notes

## Scope
This file tracks benchmark execution status and claim policy.

## Validation Levels
- `infra-validated`: binaries/scripts execute and emit expected files.
- `production-validated`: live model endpoints, fixed datasets, archived artifacts.

## Current Production-Validated Runs
- Publication benchmark suite:
  - `artifacts/publication/20260224T101457Z`
  - Script: `scripts/run_publication_benchmark_suite.py`
  - Includes metadata, retrieval metrics (Hit@k/MRR/nDCG), confidence intervals, and serving baseline deltas.
- Proposal-alignment evaluation:
  - `artifacts/proposal/20260224T103522Z`
  - Script: `scripts/run_proposal_alignment_eval.py`
  - Includes context relevance, persona consistency, naturalness, baseline comparisons, and error analysis.

## Claim Policy
- No superiority claim without corresponding artifact JSON/markdown outputs and hardware/model metadata.
- Mark every statement as one of:
  - `measured` (artifact-backed),
  - `projected` (reasoned but unmeasured),
  - `planned` (not yet implemented).


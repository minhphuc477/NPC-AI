# NPC AI Project Status

## Snapshot Date
2026-02-24

## Completed
- C++ runtime scaffold with retrieval, graph, and generation integration.
- Benchmark harnesses updated for deterministic initialization and configurable paths.
- Kaggle notebook hardened with graceful fallbacks.
- Placeholder training-data paths replaced with deterministic outputs.
- ToolRegistry built-ins moved to provider-aware execution with explicit unavailable/simulated modes.
- Added `NPC_USE_USEARCH` toggle and fallback `VectorStore` backend for environments where usearch fails to build.
- Windows fallback build validated for core benchmark targets (`bench_engine`, `ablation_suite`, `bench_retrieval`).
- Proposal-specific evaluation pipeline added:
  - `scripts/run_proposal_alignment_eval.py`
  - `data/proposal_eval_scenarios.jsonl`
  - artifacts at `artifacts/proposal/20260224T103522Z`.
- UE5 context extraction upgraded beyond placeholder weather field:
  - weather tag inference, heard-sound extraction, synthesized recent events.

## In Progress
- Persona robustness improvements for `elara-npc:latest` (reduce template/meta responses).
- BERTScore dependency fix for proposal evaluation (`triton/torch` compatibility issue).

## Blockers
- `bert_score` stack import failure in this Python environment due `triton` incompatibility.

## Claim Policy
All performance/quality claims must indicate whether they are mock-validated or production-validated.


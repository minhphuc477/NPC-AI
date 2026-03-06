# Proposal Alignment Matrix (Current)

## Scope
This document checks whether the current implementation and artifacts satisfy `docs/proposal.txt`.

## Source of Truth
- Proposal text: `docs/proposal.txt`
- Architecture: `docs/ARCHITECTURE.md`
- Latest proposal run: `artifacts/proposal/20260302T182844Z`
- Latest publication run: `artifacts/publication/20260302T191131Z`
- Strict gate report: `artifacts/proposal/20260302T182844Z/quality_gate_report.json`

## Objective-by-Objective Status

| Objective | Requirement from Proposal | Status | Evidence |
|---|---|---|---|
| RO1 | Define research gap (naturalness + persona + dynamic context) | Pass | `docs/DRAFT_PAPER.md` Sections 1-2 |
| RO2 | UE5-integrated dynamic state extraction and runtime prompting | Pass | `ue5/Source/NPCDialogue/Private/NPCContextExtractor.cpp`, `cpp/src/NPCInference.cpp`, `core/prompt_builder.py` |
| RO3 | Fine-tuned local LLM path for NPC persona behavior | Pass | `scripts/train_qlora.py`, `scripts/inference_adapter.py`, `cpp/src/ResponseController.cpp` |
| RO4 | Scientific comparison versus baselines under controlled scenarios | Pass | `artifacts/proposal/20260302T182844Z/paired_delta_significance.json`, `.../summary.json` |
| RO5 | Human/quantitative evaluation with reproducible artifacts | Pass | `artifacts/proposal/20260302T182844Z/human_eval_summary.json`, `artifacts/publication/20260302T191131Z/report.md` |

## Critical Quantitative Checks
From `quality_gate_report.json` (strict mode):
- Overall gate: **PASS** (`overall_pass=true`)
- Scenario coverage: **112** (required >= 100)
- Controlled vs raw significant improvements:
  - context relevance: +0.2131, `p=0.0`
  - persona consistency: +0.2150, `p=0.0`
  - naturalness: +0.1158, `p=0.0`
  - overall quality: +0.1808, `p=0.0`
- External wins threshold: pass for both baselines (`10/12` significant-positive metrics each)
- Human-eval thresholds:
  - row count: 324 (required >= 300)
  - mean pairwise kappa: 0.5329 (required >= 0.2)
  - soft win rate: 0.7315 vs `phi3:mini`, 0.6806 vs `phi3:latest` (required >= 0.55)

## Publication-Readiness Checks
Also passed in strict gate:
1. Non-mock artifacts + hardware/model metadata.
2. Retrieval metrics (Hit@k/MRR/nDCG) with CIs.
3. Ablation deltas and serving CIs.
4. Prompt parity for external serving comparison.
5. Security benchmark thresholds (ASR reduction and guarded ASR cap).

## Remaining Non-Blocking Gaps
These do not fail proposal quality gate, but matter for stronger paper claims:
1. Serving efficiency superiority is not shown vs lightweight baseline.
2. Core publication retrieval set is still small (10 queries), although reranker training coverage is much larger (3360 pairs).
3. External comparisons are aligned benchmark replications, not full paper-protocol reproductions.

## Verdict
With the latest artifacts (`20260302T182844Z`, `20260302T191131Z`), the project satisfies the defined proposal-quality bar and is publication-ready for claims focused on context grounding and robustness.

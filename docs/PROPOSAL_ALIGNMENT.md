# Proposal Alignment Matrix

## Scope
Trace implementation and evidence against `docs/proposal.txt` research objectives (RO1-RO5).

## Source of Truth
- Proposal: `docs/proposal.txt`
- Architecture: `docs/ARCHITECTURE.md`
- Latest proposal artifact run: `artifacts/proposal/20260224T175344Z`

## RO1: Literature Framing and Gap Definition
Status: Implemented

Evidence:
- `docs/proposal.txt`
- `docs/INDUSTRY_PUBLICATION_JUDGMENT.md`

## RO2: Integrated UE5 <-> Local LLM Architecture
Status: Implemented

Evidence:
- `docs/ARCHITECTURE.md`
- `ue5/Source/NPCDialogue/Public/NPCContextExtractor.h`
- `ue5/Source/NPCDialogue/Private/NPCContextExtractor.cpp`
- `core/prompt_builder.py`

## RO3: Persona-Tuned Prototype with Local Inference
Status: Implemented

Evidence:
- `scripts/train_qlora.py`
- `scripts/inference_adapter.py`
- `scripts/run_proposal_alignment_eval.py`
- `scripts/run_publication_benchmark_suite.py`

## RO4: Dynamic Context Method (>= 3 Context Types)
Status: Implemented

Implemented context signals:
- behavior/blackboard state
- spatial location and zone context
- nearby entities/perception events
- recent event summary

Evidence:
- `ue5/Source/NPCDialogue/Private/NPCContextExtractor.cpp`

## RO5: Scientific Evaluation Against Baselines
Status: Implemented

Pipeline:
- `scripts/run_proposal_alignment_eval.py`
- `scripts/run_proposal_alignment_eval_batched.py`
- `scripts/run_llm_multirater_campaign.py`
- `scripts/attach_human_eval_to_run.py`
- `scripts/proposal_quality_gate.py`
- Expanded scenario set: `data/proposal_eval_scenarios_large.jsonl` (112 scenarios)

Run summary (artifact `20260224T175344Z`):
- `proposed_contextual_controlled` vs `proposed_contextual`:
  - context relevance: `+0.2049`, 95% CI `(0.1837, 0.2244)`, `p<0.001`
  - persona consistency: `+0.0787`, 95% CI `(0.0495, 0.1070)`, `p<0.001`
  - naturalness: `+0.0885`, 95% CI `(0.0693, 0.1074)`, `p<0.001`
  - BERTScore F1: `+0.0342`, 95% CI `(0.0240, 0.0444)`, `p<0.001`
  - overall quality: `+0.1284`, 95% CI `(0.1139, 0.1428)`, `p<0.001`

- `proposed_contextual_controlled` vs external baseline `phi3:mini` (no context):
  - positive and significant on `10/12` metrics.

- `proposed_contextual_controlled` vs external baseline `phi3:latest` (no context):
  - positive and significant on `11/12` metrics.

## Current Gaps
1. Distinct-1 remains weaker than some external baselines.
2. Serving efficiency superiority is still not supported on current quality-normalized frontier.
3. External comparison remains same-prompt/same-dataset, not full paper-protocol replication.

## Reproduce
```bash
python scripts/generate_proposal_scenarios_large.py --variants-per-base 14
python scripts/run_proposal_alignment_eval_batched.py --scenarios data/proposal_eval_scenarios_large.jsonl --batch-size 28 --repeats 1 --max-tokens 80 --temperature 0.2 --baseline-models "phi3:latest"
python scripts/run_kaggle_full_results.py --host http://127.0.0.1:11434
```

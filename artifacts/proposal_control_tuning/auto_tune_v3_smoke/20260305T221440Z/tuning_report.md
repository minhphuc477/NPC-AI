# Control Architecture Tuning Report

- Run root: `artifacts\proposal_control_tuning\auto_tune_v3_smoke\20260305T221440Z`
- Train scenarios: `70`
- Validation scenarios: `42`
- Trials: `3`

## Recommended Candidate
- Trial index: `2`
- Overrides file: `artifacts\proposal_control_tuning\auto_tune_v3_smoke\20260305T221440Z\recommended_overrides.json`
- Train objective: `-0.1216`
- Validation objective: `-0.0901`

## Overrides
```json
{
  "min_context_coverage": 0.31,
  "min_persona_coverage": 0.16,
  "relaxed_context_coverage": 0.16,
  "relaxed_persona_coverage": 0.08,
  "relaxed_candidate_score": 0.42,
  "adaptive_candidate_score": 0.34,
  "rewrite_candidates": 2,
  "adaptive_high_confidence_rewrites": 1,
  "adaptive_mid_confidence_rewrites": 1,
  "adaptive_low_confidence_rewrites": 2,
  "low_confidence_retry_requires_gain": true,
  "low_confidence_retry_min_score_gain": 0.015,
  "low_confidence_retry_min_coverage_gain": 0.03,
  "intent_risk_adaptation_enabled": false,
  "latency_adaptation_enabled": true,
  "latency_relax_start_pressure": 0.55,
  "latency_relax_max_delta": 0.08
}
```

## Full 112 Command
```bash
C:\Python313\python.exe F:\NPC AI\scripts\run_proposal_alignment_eval_batched.py --host http://127.0.0.1:11434 --candidate-model elara-npc:latest --baseline-model phi3:mini --scenarios data\proposal_eval_scenarios_112_diverse.jsonl --batch-size 28 --repeats 1 --max-tokens 72 --temperature 0.2 --control-alt-profile custom --control-alt-arm-id proposed_contextual_controlled_tuned --control-alt-overrides-file artifacts\proposal_control_tuning\auto_tune_v3_smoke\20260305T221440Z\recommended_overrides.json --output-root artifacts\proposal_control_tuning\auto_tune_v3_smoke\20260305T221440Z\full112_recommended
```
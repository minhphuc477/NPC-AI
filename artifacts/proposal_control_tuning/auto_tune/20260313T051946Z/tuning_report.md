# Control Architecture Tuning Report

- Run root: `artifacts\proposal_control_tuning\auto_tune\20260313T051946Z`
- Train scenarios: `70`
- Validation scenarios: `42`
- Trials: `10`

## Recommended Candidate
- Trial index: `7`
- Overrides file: `artifacts\proposal_control_tuning\auto_tune\20260313T051946Z\recommended_overrides.json`
- Train objective: `-0.0212`
- Validation objective: `0.0433`

## Overrides
```json
{
  "min_context_coverage": 0.286,
  "min_persona_coverage": 0.161,
  "relaxed_context_coverage": 0.168,
  "relaxed_persona_coverage": 0.106,
  "relaxed_candidate_score": 0.48,
  "adaptive_candidate_score": 0.316,
  "adaptive_context_coverage": 0.15,
  "adaptive_persona_coverage": 0.105,
  "adaptive_high_confidence_rewrites": 1,
  "adaptive_mid_confidence_rewrites": 2,
  "adaptive_low_confidence_rewrites": 4,
  "low_confidence_retry_requires_gain": true,
  "low_confidence_retry_min_score_gain": 0.014,
  "low_confidence_retry_min_coverage_gain": 0.041,
  "early_stop_score": 0.684,
  "intent_risk_adaptation_enabled": true,
  "latency_adaptation_enabled": true,
  "latency_relax_start_pressure": 0.59,
  "latency_relax_max_delta": 0.104,
  "low_risk_context_relax": 0.052,
  "low_risk_persona_relax": 0.043,
  "low_risk_candidate_score_relax": 0.037,
  "high_risk_context_tighten": 0.017,
  "high_risk_persona_tighten": 0.009,
  "high_risk_candidate_score_tighten": 0.006,
  "intent_focused_context_enabled": true,
  "intent_focus_min_keep": 3,
  "intent_focus_keep_ratio_low": 0.471,
  "intent_focus_keep_ratio_medium": 0.633,
  "intent_focus_keep_ratio_high": 0.922,
  "intent_focus_min_relevance": 0.135,
  "near_pass_enabled": false,
  "near_pass_max_context_gap": 0.029,
  "near_pass_max_persona_gap": 0.046,
  "near_pass_score_floor": 0.418,
  "near_pass_block_high_risk": true
}
```

## Full 112 Command
```bash
F:\NPC AI\.venv\Scripts\python.exe F:\NPC AI\scripts\run_proposal_alignment_eval_batched.py --host http://127.0.0.1:11434 --candidate-model elara-npc:latest --baseline-model phi3:mini --scenarios data\proposal_eval_scenarios_112_diverse.jsonl --batch-size 28 --repeats 1 --max-tokens 72 --temperature 0.15 --control-alt-profile custom --control-alt-arm-id proposed_contextual_controlled_tuned --control-alt-overrides-file artifacts\proposal_control_tuning\auto_tune\20260313T051946Z\recommended_overrides.json --output-root artifacts\proposal_control_tuning\auto_tune\20260313T051946Z\full112_recommended
```
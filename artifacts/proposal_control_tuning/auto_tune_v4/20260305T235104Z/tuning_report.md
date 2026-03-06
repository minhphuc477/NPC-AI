# Control Architecture Tuning Report

- Run root: `artifacts\proposal_control_tuning\auto_tune_v4\20260305T235104Z`
- Train scenarios: `70`
- Validation scenarios: `42`
- Trials: `5`

## Recommended Candidate
- Trial index: `4`
- Overrides file: `artifacts\proposal_control_tuning\auto_tune_v4\20260305T235104Z\recommended_overrides.json`
- Train objective: `-0.0160`
- Validation objective: `-0.0994`

## Overrides
```json
{
  "min_context_coverage": 0.326,
  "min_persona_coverage": 0.239,
  "relaxed_context_coverage": 0.206,
  "relaxed_persona_coverage": 0.081,
  "relaxed_candidate_score": 0.406,
  "adaptive_candidate_score": 0.403,
  "adaptive_context_coverage": 0.148,
  "adaptive_persona_coverage": 0.144,
  "adaptive_high_confidence_rewrites": 1,
  "adaptive_mid_confidence_rewrites": 3,
  "adaptive_low_confidence_rewrites": 2,
  "low_confidence_retry_requires_gain": true,
  "low_confidence_retry_min_score_gain": 0.023,
  "low_confidence_retry_min_coverage_gain": 0.017,
  "early_stop_score": 0.73,
  "intent_risk_adaptation_enabled": true,
  "latency_adaptation_enabled": true,
  "latency_relax_start_pressure": 0.605,
  "latency_relax_max_delta": 0.113,
  "low_risk_context_relax": 0.057,
  "low_risk_persona_relax": 0.028,
  "low_risk_candidate_score_relax": 0.022,
  "high_risk_context_tighten": 0.027,
  "high_risk_persona_tighten": 0.026,
  "high_risk_candidate_score_tighten": 0.012,
  "intent_focused_context_enabled": false,
  "intent_focus_min_keep": 4,
  "intent_focus_keep_ratio_low": 0.423,
  "intent_focus_keep_ratio_medium": 0.548,
  "intent_focus_keep_ratio_high": 0.922,
  "intent_focus_min_relevance": 0.197
}
```

## Full 112 Command
```bash
C:\Python313\python.exe F:\NPC AI\scripts\run_proposal_alignment_eval_batched.py --host http://127.0.0.1:11434 --candidate-model elara-npc:latest --baseline-model phi3:mini --scenarios data\proposal_eval_scenarios_112_diverse.jsonl --batch-size 28 --repeats 1 --max-tokens 64 --temperature 0.2 --control-alt-profile custom --control-alt-arm-id proposed_contextual_controlled_tuned --control-alt-overrides-file artifacts\proposal_control_tuning\auto_tune_v4\20260305T235104Z\recommended_overrides.json --output-root artifacts\proposal_control_tuning\auto_tune_v4\20260305T235104Z\full112_recommended
```
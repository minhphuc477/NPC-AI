# Control Architecture Tuning Report

- Run root: `artifacts\proposal_control_tuning\auto_tune\20260305T203842Z`
- Train scenarios: `70`
- Validation scenarios: `42`
- Trials: `5`

## Recommended Candidate
- Trial index: `3`
- Overrides file: `artifacts\proposal_control_tuning\auto_tune\20260305T203842Z\recommended_overrides.json`
- Train objective: `-0.0401`
- Validation objective: `-0.1190`

## Overrides
```json
{
  "min_context_coverage": 0.351,
  "min_persona_coverage": 0.177,
  "relaxed_context_coverage": 0.211,
  "relaxed_persona_coverage": 0.138,
  "relaxed_candidate_score": 0.497,
  "adaptive_candidate_score": 0.399,
  "adaptive_context_coverage": 0.151,
  "adaptive_persona_coverage": 0.084,
  "adaptive_high_confidence_rewrites": 1,
  "adaptive_mid_confidence_rewrites": 1,
  "adaptive_low_confidence_rewrites": 4,
  "low_confidence_retry_requires_gain": true,
  "low_confidence_retry_min_score_gain": 0.017,
  "low_confidence_retry_min_coverage_gain": 0.026,
  "early_stop_score": 0.725
}
```

## Full 112 Command
```bash
C:\Python313\python.exe F:\NPC AI\scripts\run_proposal_alignment_eval_batched.py --host http://127.0.0.1:11434 --candidate-model elara-npc:latest --baseline-model phi3:mini --scenarios data\proposal_eval_scenarios_112_diverse.jsonl --batch-size 28 --repeats 1 --max-tokens 72 --temperature 0.2 --control-alt-profile custom --control-alt-arm-id proposed_contextual_controlled_tuned --control-alt-overrides-file artifacts\proposal_control_tuning\auto_tune\20260305T203842Z\recommended_overrides.json --output-root artifacts\proposal_control_tuning\auto_tune\20260305T203842Z\full112_recommended
```
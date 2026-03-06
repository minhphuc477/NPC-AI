# Control Architecture Tuning Report

- Run root: `artifacts\proposal_control_tuning\auto_tune_v2b\20260305T210504Z`
- Train scenarios: `70`
- Validation scenarios: `42`
- Trials: `4`

## Recommended Candidate
- Trial index: `0`
- Overrides file: `artifacts\proposal_control_tuning\auto_tune_v2b\20260305T210504Z\recommended_overrides.json`
- Train objective: `-0.0409`
- Validation objective: `-0.0631`

## Overrides
```json
{}
```

## Full 112 Command
```bash
C:\Python313\python.exe F:\NPC AI\scripts\run_proposal_alignment_eval_batched.py --host http://127.0.0.1:11434 --candidate-model elara-npc:latest --baseline-model phi3:mini --scenarios data\proposal_eval_scenarios_112_diverse.jsonl --batch-size 28 --repeats 1 --max-tokens 72 --temperature 0.2 --control-alt-profile custom --control-alt-arm-id proposed_contextual_controlled_tuned --control-alt-overrides-file artifacts\proposal_control_tuning\auto_tune_v2b\20260305T210504Z\recommended_overrides.json --output-root artifacts\proposal_control_tuning\auto_tune_v2b\20260305T210504Z\full112_recommended
```
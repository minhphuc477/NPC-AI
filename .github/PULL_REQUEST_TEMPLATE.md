## Summary
Add small integration demo: client script, subset dataset, and tests for BD-NSCA.

## Changes
- `scripts/call_bd_nsca.py` (new)
- `data/sample_bilingual_subset.jsonl` (new)
- `tests/test_client_bdnsca.py` (new)

## Acceptance
- [ ] Client script runs against a local server
- [ ] Client tests pass locally

## How to test locally
1. Run server (GDSearch): `uvicorn integration.inference_server:app --reload --port 8000`
2. Run: `python scripts/call_bd_nsca.py --context "guard the plaza"`
3. Run tests: `pytest tests/test_client_bdnsca.py -q`

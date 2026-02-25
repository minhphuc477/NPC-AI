# BD-NSCA Quickstart (NPC AI)

This repo contains a tiny client and dataset subset to demo the BD-NSCA inference server.

Run a local demo:
1. Start GDSearch server: `uvicorn integration.inference_server:app --reload --port 8000`
2. In this repo: `python scripts/call_bd_nsca.py --context "suspicious footprints near alley"`

Run tests:
- `pytest tests/test_client_bdnsca.py -q`

Notes: the dataset subset is intentionally small for fast demonstrations.

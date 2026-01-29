"""Simple CLI to call a BD-NSCA inference server and print actions (integration demo)."""
import argparse
import json
from typing import Dict
from inference_adapter import ModelClient, GenerateRequest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--context", required=True)
    p.add_argument("--id", default="demo-1")
    p.add_argument("--scenario", default="patrol")
    p.add_argument("--lang", default="vi")
    args = p.parse_args()

    client = ModelClient()
    req = GenerateRequest(id=args.id, scenario=args.scenario, context=args.context, agent_state={"health":100, "position": {"x":0, "y":0}}, lang=args.lang)
    resp = client.generate(req)
    print(json.dumps(resp, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Simple CLI to call BD-NSCA inference and print one response."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Dict

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
inference_adapter = importlib.import_module("inference_adapter")
BDNSCAInference = getattr(inference_adapter, "BDNSCAInference")
InferenceConfig = getattr(inference_adapter, "InferenceConfig")


def parse_context(raw: str) -> Dict[str, str]:
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = {}
    if isinstance(parsed, dict):
        return {str(k): str(v) for k, v in parsed.items()}
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Call local BD-NSCA inference adapter.")
    parser.add_argument("--persona", default="A city guard who is strict but fair.")
    parser.add_argument("--plot", default="The player arrives at the city gate.")
    parser.add_argument("--context", default="{}", help="JSON object string for dynamic context.")
    parser.add_argument("--player-input", required=True)
    parser.add_argument("--npc-name", default="Guard Captain")
    parser.add_argument("--lang", default="vi")
    parser.add_argument("--model", default="phi3:mini")
    args = parser.parse_args()

    context = parse_context(args.context)
    cfg = InferenceConfig(model=str(args.model))
    client = BDNSCAInference(config=cfg)
    response = client.generate(
        persona=str(args.persona),
        plot=str(args.plot),
        context=context,
        player_input=str(args.player_input),
        npc_name=str(args.npc_name),
        language=str(args.lang),
    )
    payload: Dict[str, Any] = {
        "ok": True,
        "model": str(args.model),
        "npc_name": str(args.npc_name),
        "response": response,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

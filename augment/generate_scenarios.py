"""CLI for generating mock scenarios for augmentation.

Usage: python -m augment.generate_scenarios --n 10 --teacher mock
"""
import argparse
import json
from pathlib import Path
import random


def mock_generate(n=10):
    examples = []
    for i in range(n):
        ex = {
            "id": str(i),
            "prompt": f"NPC greets player {i}",
            "response": random.choice(["Xin chao", "Chao ban", "Chào mừng"]) 
        }
        examples.append(ex)
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--teacher", type=str, default="mock")
    parser.add_argument("--out", type=str, default="data/generated/scenarios.jsonl")
    args = parser.parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.teacher == "mock":
        examples = mock_generate(args.n)
    else:
        raise NotImplementedError("Only mock teacher is implemented in this stub.")
    with out_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Wrote {len(examples)} examples to {out_path}")


if __name__ == "__main__":
    main()
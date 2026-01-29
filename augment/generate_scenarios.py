"""CLI for generating mock scenarios for augmentation.

Usage: python -m augment.generate_scenarios --n 10 --teacher mock
"""
import argparse
import json
from pathlib import Path
import random
import os
from datetime import datetime


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
    parser.add_argument("--mock", action="store_true", help="Use mock generator even if a Groq key is present")
    parser.add_argument("--out", type=str, default="data/generated/groq_generated.jsonl")
    parser.add_argument("--model", type=str, default="llama-3.1-8b")
    args = parser.parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.teacher == "mock":
        examples = mock_generate(args.n)
        meta = {"model": "mock", "timestamp": None, "cost_estimate": 0}
    elif args.teacher == "groq":
        # allow explicit mock flag for offline testing
        if args.mock or not os.environ.get('GROQ_API_KEY'):
            examples = mock_generate(args.n)
            model_used = "mock"
        else:
            from scripts.colab_helpers import batch_generate_with_groq
            prompts = [f"NPC prompt {i}" for i in range(args.n)]
            outputs = batch_generate_with_groq(prompts, model_id=args.model)
            examples = []
            for i, o in enumerate(outputs):
                examples.append({"id": str(i), "prompt": prompts[i], "response": o})
            model_used = args.model
        meta = {"model": model_used, "timestamp": datetime.utcnow().isoformat(), "cost_estimate": "TBD"}
    else:
        raise NotImplementedError("Only 'mock' and 'groq' teachers are implemented.")

    # write outputs
    with out_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # write metadata next to outputs
    meta_path = out_path.parent / (out_path.name.replace('.jsonl', '') + "_meta.json")
    meta.update({"n": len(examples)})
    with meta_path.open("w", encoding="utf-8") as mf:
        json.dump(meta, mf, ensure_ascii=False, indent=2)

    print(f"Wrote {len(examples)} examples to {out_path} and metadata to {meta_path}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Generate DPO (Direct Preference Optimization) training data.
Creates pairs of (chosen, rejected) responses to teach the model style and persona.

Strategy:
- Chosen: The correct, persona-aligned response (from verified dataset)
- Rejected: 
    1. Response from a different persona (e.g., Merchant speaking like Gatekeeper)
    2. Generic/Bland response (lacking personality)
    3. Rude/Inappropriate response (if available)

Usage:
    python scripts/generate_dpo_data.py --input data/train_combined.jsonl --output data/train_dpo.jsonl
"""
import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

def load_jsonl(path: str) -> List[Dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def save_jsonl(data: List[Dict], path: str):
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def generate_rejected_response(sample: Dict, all_samples: List[Dict]) -> str:
    """Generate a rejected response for DPO."""
    strategy = random.choice(['wrong_persona', 'generic', 'wrong_language'])
    
    # 1. Wrong Persona: Pick a response from a different NPC ID
    if strategy == 'wrong_persona':
        other_samples = [s for s in all_samples if s['metadata']['npc_id'] != sample['metadata']['npc_id']]
        if other_samples:
            return random.choice(other_samples)['completion']
    
    # 2. Generic/Bland
    if strategy == 'generic':
        if sample['metadata'].get('language') == 'vi':
            return random.choice([
                "Tôi không biết.",
                "Xin chào.",
                "Thời tiết hôm nay đẹp nhỉ.",
                "Bạn muốn gì?",
                "Tôi là một NPC."
            ])
        else:
            return random.choice([
                "I don't know.",
                "Hello.",
                "Nice weather today.",
                "What do you want?",
                "I am an NPC."
            ])
            
    # Fallback to random sample
    return random.choice(all_samples)['completion']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/train_combined.jsonl")
    parser.add_argument("--output", default="data/train_dpo.jsonl")
    args = parser.parse_args()
    
    data = load_jsonl(args.input)
    dpo_data = []
    
    print(f"Generating DPO pairs from {len(data)} samples...")
    
    for sample in data:
        # Prompt already contains <|system|>...<|user|>...<|assistant|>
        # So we can use it directly as the DPO prompt
        
        dpo_sample = {
            "prompt": sample['prompt'],
            "chosen": sample['completion'],
            "rejected": generate_rejected_response(sample, data),
            "metadata": sample['metadata']
        }
        dpo_data.append(dpo_sample)
        
    save_jsonl(dpo_data, args.output)
    print(f"Saved {len(dpo_data)} DPO pairs to {args.output}")

if __name__ == "__main__":
    main()

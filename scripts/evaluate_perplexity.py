#!/usr/bin/env python3
"""
Evaluate Perplexity of Fine-Tuned Model

Loads the base model and calculates perplexity on a specified test dataset (json or jsonl).
"""

import argparse
import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
        return []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="gpt2", help="Path or name of base model.")
    parser.add_argument("--adapter", type=str, default="outputs/npc_model", help="Path to LoRA adapter. If 'none', evaluates base model.")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test dataset (json format).")
    parser.add_argument("--max_samples", type=int, default=10, help="Maximum number of samples to evaluate.")
    args = parser.parse_args()

    print(f"Loading Base Model: {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    
    # Load explicitly on CPU to bypass all CUDA/Triton/FlashAttention extension issues
    dtype = torch.float32
    print("Loading model on CPU with float32 (this will be slow but reliable)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map="cpu",
        trust_remote_code=True
    )
    
    model.eval()

    print(f"Loading test data from {args.test_data}...")
    try:
        data = load_json(args.test_data)
    except Exception as e:
        print(f"Error loading test data: {e}")
        return
        
    if len(data) > args.max_samples:
        print(f"Truncating {len(data)} samples to {args.max_samples} for evaluation.")
        data = data[:args.max_samples]
        
    print(f"Evaluating Perplexity on {len(data)} samples...")
    
    nlls = []
    
    for sample in tqdm(data):
        # Depending on the data format, we reconstruct the full text or prompt+completion
        text = sample.get("text", "")
        if not text:
            prompt = sample.get("prompt", "")
            completion = sample.get("completion", "")
            text = prompt + completion
            
        if not text:
            continue
            
        encodings = tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids.to(model.device)
        
        # Calculate NLL
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            # Loss is already CrossEntropy over the tokens, which is equivalent to NLL
            neg_log_likelihood = outputs.loss
            
        nlls.append(neg_log_likelihood)

    if nlls:
        avg_nll = torch.stack(nlls).mean()
        ppl = torch.exp(avg_nll).item()
        print(f"\nResults:")
        print(f"Base Model: {args.base_model}")
        print(f"Test Set:   {args.test_data}")
        print(f"Samples:    {len(nlls)}")
        print(f"Average NLL: {avg_nll.item():.4f}")
        print(f"Perplexity:  {ppl:.4f}")
    else:
        print("No valid samples evaluated.")

if __name__ == "__main__":
    main()

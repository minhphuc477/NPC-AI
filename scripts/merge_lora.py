
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os

def merge_lora(base_model_path, lora_path, output_path):
    print(f"Loading base model from {base_model_path}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"Loading LoRA adapters from {lora_path}...")
    model = PeftModel.from_pretrained(base_model, lora_path)
    
    print("Merging weights...")
    model = model.merge_and_unload()
    
    print(f"Saving merged model to {output_path}...")
    model.save_pretrained(output_path)
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--lora-path", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()
    
    merge_lora(args.base_model, args.lora_path, args.output_path)

#!/usr/bin/env python3
"""Test inference with trained QLoRA adapter."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import argparse

def load_model(base_model, adapter_path):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    logger.info(f"Loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    if adapter_path:
        logger.info(f"Loading adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_new_tokens=200):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--adapter-path", default="outputs/adapter")
    parser.add_argument("--prompt", default="<|user|>Xin chao<|end|>\n<|assistant|>")
    args = parser.parse_args()

    logger.info("Loading model...")
    model, tokenizer = load_model(args.base_model, args.adapter_path)
    
    response = generate_response(model, tokenizer, args.prompt)
    print("\n=== GENERATED RESPONSE ===\n")
    print(response)


#!/usr/bin/env python3
"""Generate predictions from trained model for evaluation."""
import json
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
ADAPTER_PATH = "outputs/adapter"
TEST_DATA = "data/test.jsonl"
PREDICTIONS_FILE = "data/predictions.jsonl"


def load_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    logger.info(f"Loading base model: {BASE_MODEL}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    logger.info(f"Loading adapter: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    return model, tokenizer


def generate_response(model, tokenizer, prompt, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False,  # Fix DynamicCache compatibility
        )
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated = full_response[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):]
    return generated.strip()


def main():
    parser = argparse.ArgumentParser(description="Generate predictions for evaluation")
    parser.add_argument("--base_model", default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--adapter_path", default="outputs/adapter")
    parser.add_argument("--test_data", default="data/test.jsonl")
    parser.add_argument("--output", default="data/predictions.jsonl")
    args = parser.parse_args()
    
    global BASE_MODEL, ADAPTER_PATH
    BASE_MODEL = args.base_model
    ADAPTER_PATH = args.adapter_path
    
    model, tokenizer = load_model()
    
    predictions = []
    if not Path(args.test_data).exists():
        logger.warning(f"Test data not found: {args.test_data}")
        return

    with open(args.test_data, "r", encoding="utf-8") as f:
        test_samples = [json.loads(line) for line in f if line.strip()]
    

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Clear file first
    with open(output_path, "w", encoding="utf-8") as f:
        pass
        
    logger.info(f"Generating predictions for {len(test_samples)} samples...")
    
    for i, sample in enumerate(test_samples):
        prompt = sample["prompt"]
        reference = sample.get("completion", "")
        
        prediction = generate_response(model, tokenizer, prompt)
        
        pred_obj = {
            "id": sample.get("id", f"test_{i}"),
            "prompt": prompt,
            "reference": reference,
            "prediction": prediction,
            "metadata": sample.get("metadata", {})
        }
        
        predictions.append(pred_obj)
        
        # Write incrementally
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(pred_obj, ensure_ascii=False) + "\n")
        
        # Log every sample to see progress
        logger.info(f"Processed {i + 1}/{len(test_samples)}")
    
    if predictions:
        print("\n=== Sample Predictions ===")
        pred = predictions[0]
        print(f"\nPrompt: {pred['prompt'][:100]}...")
        print(f"Reference: {pred['reference'][:100]}...")
        print(f"Prediction: {pred['prediction'][:100]}...")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Train DPO (Direct Preference Optimization) model.
Aligns the model with desired persona and style using preference pairs.

Usage:
    python scripts/train_dpo.py --model_name_or_path microsoft/Phi-3-mini-4k-instruct --adapter_path outputs/adapter
"""
import os
import torch
from dataclasses import dataclass, field
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--adapter_path", default=None, help="Path to SFT adapter (optional)")
    parser.add_argument("--data_path", default="data/train_dpo.jsonl")
    parser.add_argument("--output_dir", default="outputs/dpo_adapter")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing")
    args = parser.parse_args()

    # Load dataset
    dataset = load_dataset("json", data_files=args.data_path, split="train")

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load adapter if provided (continue training or base for DPO)
    if args.adapter_path:
        print(f"Loading SFT adapter from {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=True)
    else:
        # If no adapter, add new LoRA adapter
        peft_config = LoraConfig(
            r=16,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear"
        )
        model = get_peft_model(model, peft_config)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # DPO Config (replaces TrainingArguments)
    training_args = DPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=5e-5,
        logging_steps=10,
        num_train_epochs=1,
        max_steps=100,
        fp16=False,
        bf16=False,
        optim="paged_adamw_32bit",
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=args.gradient_checkpointing,
        beta=0.1,
        max_length=512,
        max_prompt_length=256,
    )

    # Initialize DPO Trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=None, # Implicit ref model
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer, # Renamed from tokenizer in recent TRL versions
    )

    print("Starting DPO training...")
    trainer.train()
    
    print(f"Saving DPO adapter to {args.output_dir}")
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()

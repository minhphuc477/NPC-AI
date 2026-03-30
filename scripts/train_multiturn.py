#!/usr/bin/env python3
"""
BD-NSCA Multi-Turn Training Script (QLoRA) - v2

Key improvement: Uses DataCollatorForCompletionOnlyLM to train ONLY on
assistant response tokens, not on system/user tokens. This prevents the
model from wasting capacity learning to predict user input.

Usage:
    python scripts/train_multiturn.py --data data/train_multiturn.jsonl --output-dir storage/outputs/adapter_multiturn
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import warnings
import inspect
import importlib

# Suppress flash-attention warnings (not installed but optional)
warnings.filterwarnings("ignore", message=".*flash-attention.*")
warnings.filterwarnings("ignore", message=".*flash_attn.*")

import torch
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

# Suppress logging warnings from transformers model code BEFORE importing transformers
logging.getLogger("transformers_modules").setLevel(logging.ERROR)

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Response template for Phi-3 chat format
# Loss is ONLY computed on tokens AFTER this marker
RESPONSE_TEMPLATE = "<|assistant|>" + "\n"


class CompletionOnlyCollator:
    """Mask labels before the assistant marker for completion-only loss."""

    def __init__(self, tokenizer: Any, response_template: str):
        self.tokenizer = tokenizer
        self.response_token_ids = tokenizer.encode(response_template, add_special_tokens=False)
        self.base_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    @staticmethod
    def _find_subsequence(sequence: List[int], pattern: List[int]) -> int:
        if not pattern or len(pattern) > len(sequence):
            return -1
        window = len(pattern)
        for idx in range(0, len(sequence) - window + 1):
            if sequence[idx : idx + window] == pattern:
                return idx
        return -1

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = self.base_collator(features)
        input_ids = batch.get("input_ids")
        labels = batch.get("labels")
        if input_ids is None or labels is None:
            return batch

        for row_idx in range(int(input_ids.shape[0])):
            tokens = [int(x) for x in input_ids[row_idx].tolist()]
            start = self._find_subsequence(tokens, self.response_token_ids)
            if start < 0:
                labels[row_idx, :] = -100
                continue
            response_start = start + len(self.response_token_ids)
            labels[row_idx, :response_start] = -100
        batch["labels"] = labels
        return batch


@dataclass
class TrainingConfig:
    # Model
    base_model: str = "microsoft/Phi-3-mini-4k-instruct"
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = False

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # Training
    num_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_seq_length: int = 2048  # Longer for multi-turn

    # Output
    output_dir: str = "storage/outputs/adapter_multiturn"
    logging_steps: int = 10
    save_steps: int = 100
    max_steps: int = -1

    def to_dict(self) -> Dict:
        return {k: str(v) for k, v in self.__dict__.items()}


def load_dataset(data_path: str) -> List[Dict]:
    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    logger.info("Loaded %d samples from %s", len(samples), data_path)
    return samples


def format_for_training(samples: List[Dict]) -> List[Dict]:
    formatted = []
    for sample in samples:
        text = sample.get("text", sample.get("prompt", "") + sample.get("completion", ""))
        formatted.append({"text": text})
    return formatted


def run_training(config: TrainingConfig, data_path: str):
    peft_mod = importlib.import_module("peft")
    LoraConfig = getattr(peft_mod, "LoraConfig")
    prepare_model_for_kbit_training = getattr(peft_mod, "prepare_model_for_kbit_training")
    trl_sft_mod = importlib.import_module("trl.trainer.sft_trainer")
    SFTTrainer = getattr(trl_sft_mod, "SFTTrainer")
    trl_cfg_mod = importlib.import_module("trl.trainer.sft_config")
    SFTConfig = getattr(trl_cfg_mod, "SFTConfig")
    datasets_mod = importlib.import_module("datasets")
    Dataset = getattr(datasets_mod, "Dataset")

    # Check CUDA
    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Training will be slow!")

    # Load dataset
    raw_samples = load_dataset(data_path)
    formatted_samples = format_for_training(raw_samples)
    dataset = Dataset.from_list(formatted_samples)

    # Config setup
    compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.use_4bit,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=config.use_nested_quant,
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        r=config.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.target_modules,
    )

    # --- KEY FIX: Use DataCollatorForCompletionOnlyLM ---
    # This masks loss on everything EXCEPT assistant responses.
    # The model only learns to generate NPC dialogue, not to predict
    # system prompts or player input.
    collator = CompletionOnlyCollator(
        response_template=RESPONSE_TEMPLATE,
        tokenizer=tokenizer,
    )

    sft_config = SFTConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        optim="paged_adamw_32bit",
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        learning_rate=config.learning_rate,
        fp16=False,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="none",
        max_steps=config.max_steps,
        max_length=config.max_seq_length,
        dataset_text_field="text",
    )

    def formatting_func(example):
        return example["text"]

    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "train_dataset": dataset,
        "peft_config": peft_config,
        "args": sft_config,
        "formatting_func": formatting_func,
        "data_collator": collator,
    }
    init_params = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in init_params:
        trainer_kwargs["processing_class"] = tokenizer
    if "tokenizer" in init_params:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = SFTTrainer(**trainer_kwargs)

    logger.info("Starting multi-turn training with completion-only loss masking...")
    logger.info("Response template: %s", repr(RESPONSE_TEMPLATE))
    trainer.train()

    logger.info("Saving adapter to %s", config.output_dir)
    saved_model = trainer.model
    if saved_model is None:
        raise RuntimeError("Trainer returned no model instance after training.")
    saved_model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", default="storage/outputs/adapter_multiturn")
    parser.add_argument("--base-model", default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    args = parser.parse_args()

    config = TrainingConfig(
        base_model=args.base_model,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
    )

    run_training(config, args.data)


if __name__ == "__main__":
    main()

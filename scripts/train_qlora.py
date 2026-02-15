#!/usr/bin/env python3
"""
BD-NSCA QLoRA Fine-tuning Script

Complete implementation for fine-tuning Phi-3 Mini or Llama 3 using QLoRA.
Designed to run on Google Colab with GPU.

Usage (Colab):
    !python train_qlora.py --data data/train.jsonl --output-dir outputs/adapter --epochs 3

Usage (Dry run / local test):
    python train_qlora.py --data data/train.jsonl --output-dir outputs/adapter --dry-run
"""
from __future__ import annotations
import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for QLoRA fine-tuning."""
    # Model
    base_model: str = "microsoft/Phi-3-mini-4k-instruct"
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16" # Optimizing for Ampere+ GPUs
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = False
    
    # LoRA
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Training
    num_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    weight_decay: float = 0.001
    max_grad_norm: float = 0.3
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    max_seq_length: int = 1024 # Reduced from 2048 to save VRAM
    
    # Output
    output_dir: str = "models/phi3_npc_lora"
    logging_steps: int = 5
    save_steps: int = 50
    gradient_checkpointing: bool = True
    
    def to_dict(self) -> Dict:
        return {
            "base_model": self.base_model,
            "use_4bit": self.use_4bit,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "max_seq_length": self.max_seq_length,
        }


def load_dataset(data_path: str) -> List[Dict]:
    """Load JSONL dataset."""
    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    logger.info(f"Loaded {len(samples)} samples from {data_path}")
    return samples


def format_for_training(samples: List[Dict]) -> List[Dict]:
    """Format samples for the trainer.
    
    Converts prompt-completion pairs to the format expected by SFTTrainer.
    """
    formatted = []
    for sample in samples:
        # Combine prompt and completion into a single text field
        text = sample["prompt"] + sample["completion"]
        formatted.append({"text": text})
    return formatted


def run_training(config: TrainingConfig, data_path: str):
    """Run the actual QLoRA training.
    
    This requires GPU and the following packages:
    - transformers
    - peft
    - bitsandbytes
    - accelerate
    - trl
    - datasets
    """
    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainingArguments,
        )
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from trl import SFTTrainer
        from datasets import Dataset
    except ImportError as e:
        raise ImportError(
            f"Missing required package: {e}. "
            "Please install: pip install transformers peft bitsandbytes accelerate trl datasets"
        )
    
    # Check CUDA
    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Training will be very slow!")
    else:
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load dataset
    raw_samples = load_dataset(data_path)
    formatted_samples = format_for_training(raw_samples)
    dataset = Dataset.from_list(formatted_samples)
    
    # BitsAndBytes config for 4-bit quantization
    compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.use_4bit,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=config.use_nested_quant,
    )
    
    logger.info(f"Loading base model: {config.base_model}")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,  # Explicit FP16 for compatibility
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA config
    peft_config = LoraConfig(
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        r=config.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.target_modules,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        optim="paged_adamw_32bit",
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        fp16=False,  # Disable AMP to avoid BFloat16 incompatibility
        bf16=False,
        max_grad_norm=config.max_grad_norm,
        max_steps=-1,
        warmup_ratio=config.warmup_ratio,
        group_by_length=True,
        lr_scheduler_type=config.lr_scheduler_type,
        gradient_checkpointing=config.gradient_checkpointing,
        report_to="none",  # Disable wandb/tensorboard for simplicity
    )
    
    # Initialize trainer (TRL 0.27+ compatible)
    def formatting_func(example):
        return example["text"]
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_args,
        formatting_func=formatting_func,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save adapter
    logger.info(f"Saving adapter to {config.output_dir}")
    trainer.model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    
    # Save config
    config_path = Path(config.output_dir) / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    
    logger.info("Training complete!")
    return config.output_dir


def run_dry_run(config: TrainingConfig, data_path: str):
    """Validate data and config without actual training."""
    logger.info("=== DRY RUN MODE ===")
    
    # Validate data exists
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Load and validate samples
    samples = load_dataset(data_path)
    if len(samples) == 0:
        raise ValueError("Dataset is empty!")
    
    # Check sample format
    sample = samples[0]
    required_keys = ["prompt", "completion"]
    for key in required_keys:
        if key not in sample:
            raise ValueError(f"Sample missing required key: {key}")
    
    logger.info(f"✓ Dataset valid: {len(samples)} samples")
    logger.info(f"✓ Sample keys: {list(sample.keys())}")
    
    # Format check
    formatted = format_for_training(samples[:5])
    avg_len = sum(len(s["text"]) for s in formatted) / len(formatted)
    logger.info(f"✓ Average sample length: {avg_len:.0f} characters")
    
    # Config summary
    logger.info(f"✓ Base model: {config.base_model}")
    logger.info(f"✓ LoRA rank: {config.lora_r}, alpha: {config.lora_alpha}")
    logger.info(f"✓ Epochs: {config.num_epochs}, LR: {config.learning_rate}")
    logger.info(f"✓ Output: {config.output_dir}")
    
    # Check for GPU
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("⚠ No GPU detected. Training will be slow!")
    except ImportError:
        logger.warning("⚠ PyTorch not installed. Cannot check GPU.")
    
    logger.info("=== DRY RUN COMPLETE ===")
    return True


def main():
    parser = argparse.ArgumentParser(description="BD-NSCA QLoRA Fine-tuning")
    parser.add_argument("--data", required=True, help="Path to training JSONL file")
    parser.add_argument("--output-dir", default="outputs/adapter", help="Output directory for adapter")
    parser.add_argument("--base-model", default="microsoft/Phi-3-mini-4k-instruct", 
                        help="Base model (HuggingFace ID or local path)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing (saves VRAM)")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--max-seq-length", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--dry-run", action="store_true", help="Validate without training")
    args = parser.parse_args()
    
    config = TrainingConfig(
        base_model=args.base_model,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        max_seq_length=args.max_seq_length,
    )
    
    # Pass gradient checkpointing via a new attribute or modify run_training to use it
    config.gradient_checkpointing = args.gradient_checkpointing
    
    if args.dry_run:
        run_dry_run(config, args.data)
    else:
        run_training(config, args.data)


if __name__ == "__main__":
    main()

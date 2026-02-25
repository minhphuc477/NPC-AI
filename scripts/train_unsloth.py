#!/usr/bin/env python3
"""GPU-only Unsloth trainer (fallback path for CUDA notebooks)."""

from __future__ import annotations

import argparse
import json
import os

from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel
import torch


def load_json_records(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    raise ValueError("Expected a JSON array dataset.")


def train(dataset_path: str, output_dir: str, max_steps: int = 120) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("train_unsloth.py is CUDA-only. Use scripts/train_qlora.py for TPU/CPU.")

    records = load_json_records(dataset_path)
    formatted = []
    for row in records:
        prompt = str(row.get("prompt", ""))
        completion = str(row.get("completion", ""))
        if prompt and completion:
            formatted.append({"text": f"{prompt}{completion}<|end|>"})
    if not formatted:
        raise ValueError("No valid prompt/completion rows found.")

    dataset = Dataset.from_list(formatted)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Phi-3-mini-4k-instruct-bnb-4bit",
        max_seq_length=2048,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model=model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    os.makedirs(output_dir, exist_ok=True)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        args=SFTConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            max_steps=max_steps,
            learning_rate=2e-4,
            save_strategy="steps",
            save_steps=max(20, max_steps // 3),
            save_total_limit=2,
            logging_steps=1,
            optim="adamw_8bit",
            output_dir=output_dir,
            report_to="none",
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
        ),
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved adapter to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="data/npc_training_v2.json")
    parser.add_argument("--output_dir", type=str, default="outputs/npc_model")
    parser.add_argument("--max_steps", type=int, default=120)
    args = parser.parse_args()
    train(args.dataset, args.output_dir, max_steps=args.max_steps)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Preference optimization trainer (DPO) for significant behavior gains over SFT-only runs.

Expected dataset format (jsonl/json):
{
  "prompt": "...",
  "chosen": "...",
  "rejected": "...",
  "metadata": {...}  # optional
}
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _disable_torchao_if_needed() -> None:
    # Work around incompatible torchao/triton environments.
    try:
        import transformers.utils.import_utils as iu  # type: ignore

        iu._torchao_available = False
        iu._torchao_version = "0.0.0"
    except Exception:
        pass


@dataclass
class DPOConfig:
    base_model: str = "microsoft/Phi-3-mini-4k-instruct"
    reference_model: str = ""
    dataset_path: str = "artifacts/proposal/latest/preference_dataset.jsonl"
    output_dir: str = "outputs/dpo_adapter"
    accelerator: str = "auto"  # auto | cuda | cpu

    # QLoRA/LoRA
    use_4bit: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    # Optimization
    beta: float = 0.1
    learning_rate: float = 5e-6
    num_train_epochs: int = 2
    max_steps: int = -1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.05
    weight_decay: float = 0.0
    max_prompt_length: int = 1024
    max_length: int = 1536
    logging_steps: int = 10
    save_steps: int = 50
    save_total_limit: int = 2
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_model": self.base_model,
            "reference_model": self.reference_model,
            "dataset_path": self.dataset_path,
            "output_dir": self.output_dir,
            "accelerator": self.accelerator,
            "use_4bit": self.use_4bit,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "beta": self.beta,
            "learning_rate": self.learning_rate,
            "num_train_epochs": self.num_train_epochs,
            "max_steps": self.max_steps,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "warmup_ratio": self.warmup_ratio,
            "max_prompt_length": self.max_prompt_length,
            "max_length": self.max_length,
            "seed": self.seed,
        }


def detect_accelerator(requested: str) -> str:
    req = (requested or "auto").strip().lower()
    if req not in {"auto", "cuda", "cpu"}:
        raise ValueError(f"Unsupported accelerator: {requested}")
    if req != "auto":
        return req
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def ensure_preference_columns(dataset: Any) -> Any:
    required = {"prompt", "chosen", "rejected"}
    cols = set(getattr(dataset, "column_names", []))
    missing = sorted(required - cols)
    if missing:
        raise ValueError(f"Preference dataset missing columns: {missing}")
    return dataset


def resolve_target_modules(model: Any, requested: List[str]) -> List[str]:
    module_names = [name for name, _ in model.named_modules()]
    present = [t for t in requested if any(name.endswith(t) for name in module_names)]
    if present:
        return present
    fallback_groups = [
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ["c_attn", "c_proj", "c_fc"],
        ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    ]
    for group in fallback_groups:
        group_present = [t for t in group if any(name.endswith(t) for name in module_names)]
        if group_present:
            logger.warning("Requested LoRA targets unavailable, using fallback: %s", group_present)
            return group_present
    raise RuntimeError("Could not resolve LoRA target modules for this base model.")


def load_dataset(path: str) -> Any:
    from datasets import load_dataset

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")
    ds = load_dataset("json", data_files=str(p), split="train")
    return ensure_preference_columns(ds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DPO preference adapter")
    parser.add_argument("--dataset", required=True, help="Preference dataset path (.json or .jsonl)")
    parser.add_argument("--base-model", default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--reference-model", default="", help="Optional explicit reference model")
    parser.add_argument("--output-dir", default="outputs/dpo_adapter")
    parser.add_argument("--accelerator", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization on CUDA")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-acc", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = DPOConfig(
        base_model=str(args.base_model),
        reference_model=str(args.reference_model),
        dataset_path=str(args.dataset),
        output_dir=str(args.output_dir),
        accelerator=str(args.accelerator),
        use_4bit=not bool(args.no_4bit),
        lora_r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        beta=float(args.beta),
        learning_rate=float(args.lr),
        num_train_epochs=int(args.epochs),
        max_steps=int(args.max_steps),
        per_device_train_batch_size=int(args.batch_size),
        gradient_accumulation_steps=int(args.grad_acc),
        warmup_ratio=float(args.warmup_ratio),
        weight_decay=float(args.weight_decay),
        max_prompt_length=int(args.max_prompt_length),
        max_length=int(args.max_length),
        logging_steps=int(args.logging_steps),
        save_steps=int(args.save_steps),
        save_total_limit=int(args.save_total_limit),
        seed=int(args.seed),
    )

    accel = detect_accelerator(cfg.accelerator)
    logger.info("Accelerator: %s", accel)

    _disable_torchao_if_needed()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    try:
        from trl import DPOTrainer
    except Exception as exc:
        raise RuntimeError(
            "Missing TRL DPOTrainer. Install/upgrade `trl` to a version supporting DPO."
        ) from exc
    try:
        from trl import DPOConfig as TRLDPOConfig  # type: ignore
    except Exception:
        TRLDPOConfig = None

    dataset = load_dataset(cfg.dataset_path)
    logger.info("Loaded preference rows: %d", len(dataset))

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if accel == "cuda":
        model_kwargs["device_map"] = "auto"
        if cfg.use_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=False,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
    else:
        model_kwargs["torch_dtype"] = torch.float32

    logger.info("Loading policy model: %s", cfg.base_model)
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, **model_kwargs)

    if accel == "cuda" and cfg.use_4bit:
        model = prepare_model_for_kbit_training(model)

    targets = resolve_target_modules(model, cfg.target_modules)
    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    model = get_peft_model(model, lora_cfg)
    logger.info("LoRA target modules: %s", targets)

    ref_model = None
    if cfg.reference_model.strip():
        logger.info("Loading explicit reference model: %s", cfg.reference_model)
        ref_model = AutoModelForCausalLM.from_pretrained(cfg.reference_model, **model_kwargs)

    base_args = dict(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        bf16=(accel == "cuda"),
        fp16=False,
        gradient_checkpointing=True,
        report_to=[],
        seed=cfg.seed,
        remove_unused_columns=False,
    )

    if TRLDPOConfig is not None:
        train_args = TRLDPOConfig(
            **base_args,
            beta=cfg.beta,
            max_prompt_length=cfg.max_prompt_length,
            max_length=cfg.max_length,
            precompute_ref_log_probs=False,
        )
    else:
        train_args = TrainingArguments(**base_args)
        # Compatibility shim for TRL versions expecting DPOConfig-like attributes.
        compat_defaults = {
            "model_init_kwargs": None,
            "ref_model_init_kwargs": None,
            "beta": cfg.beta,
            "label_pad_token_id": -100,
            "padding_value": tokenizer.pad_token_id,
            "truncation_mode": "keep_end",
            "max_prompt_length": cfg.max_prompt_length,
            "max_length": cfg.max_length,
            "max_completion_length": max(32, cfg.max_length - cfg.max_prompt_length),
            "dataset_num_proc": None,
            "precompute_ref_log_probs": False,
            "precompute_ref_batch_size": 8,
            "disable_dropout": False,
            "generate_during_eval": False,
            "tools": None,
            "activation_offloading": False,
            "padding_free": False,
            "use_logits_to_keep": False,
        }
        for key, value in compat_defaults.items():
            if not hasattr(train_args, key):
                setattr(train_args, key, value)

    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "ref_model": ref_model,
        "args": train_args,
        "train_dataset": dataset,
    }

    init_sig = inspect.signature(DPOTrainer.__init__).parameters
    if "tokenizer" in init_sig:
        trainer_kwargs["tokenizer"] = tokenizer
    if "processing_class" in init_sig:
        trainer_kwargs["processing_class"] = tokenizer
    if "beta" in init_sig:
        trainer_kwargs["beta"] = cfg.beta
    if "max_prompt_length" in init_sig:
        trainer_kwargs["max_prompt_length"] = cfg.max_prompt_length
    if "max_length" in init_sig:
        trainer_kwargs["max_length"] = cfg.max_length

    logger.info("Initializing DPO trainer...")
    trainer = DPOTrainer(**trainer_kwargs)
    logger.info("Starting DPO training...")
    trainer.train()

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    with (Path(cfg.output_dir) / "dpo_training_config.json").open("w", encoding="utf-8") as handle:
        json.dump(cfg.to_dict(), handle, indent=2, ensure_ascii=False)

    logger.info("Saved DPO adapter/model to %s", cfg.output_dir)


if __name__ == "__main__":
    # Suppress noisy tokenizer parallelism warning in notebook environments.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()

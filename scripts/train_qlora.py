#!/usr/bin/env python3
"""
BD-NSCA LoRA/QLoRA training script with CUDA/TPU/CPU support.

Modes:
- CUDA: QLoRA (4-bit bitsandbytes) by default.
- TPU/CPU: standard LoRA (no bitsandbytes, bf16/fp32).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _in_kaggle() -> bool:
    return Path("/kaggle").exists()


def _find_in_kaggle_input(filename: str) -> str | None:
    root = Path("/kaggle/input")
    if not root.exists():
        return None
    matches = list(root.rglob(filename))
    for match in matches:
        if match.is_file():
            return str(match)
    return None


def resolve_kaggle_paths(data_path: str, output_dir: str) -> tuple[str, str]:
    """Resolve common Kaggle path issues for data/output locations."""
    resolved_data = str(data_path)
    resolved_output = str(output_dir)

    if not _in_kaggle():
        return resolved_data, resolved_output

    data = Path(data_path)
    if not data.exists():
        candidate = _find_in_kaggle_input(data.name)
        if candidate:
            logger.info("Resolved missing data path via /kaggle/input: %s", candidate)
            resolved_data = candidate

    out = Path(output_dir)
    if not out.is_absolute():
        resolved_output = str(Path("/kaggle/working") / out)
        logger.info("Using Kaggle writable output path: %s", resolved_output)

    return resolved_data, resolved_output


@dataclass
class TrainingConfig:
    base_model: str = "microsoft/Phi-3-mini-4k-instruct"
    output_dir: str = "outputs/adapter"
    accelerator: str = "auto"  # auto | cuda | tpu | cpu

    # Quantization (CUDA only)
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = False

    # LoRA
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

    # Training
    num_epochs: int = 3
    max_steps: int = -1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.001
    max_grad_norm: float = 0.3
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    max_seq_length: int = 1024
    gradient_checkpointing: bool = True
    logging_steps: int = 5
    save_steps: int = 50
    save_total_limit: int = 2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_model": self.base_model,
            "output_dir": self.output_dir,
            "accelerator": self.accelerator,
            "use_4bit": self.use_4bit,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "num_epochs": self.num_epochs,
            "max_steps": self.max_steps,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "max_seq_length": self.max_seq_length,
            "gradient_checkpointing": self.gradient_checkpointing,
        }


def _parse_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [x for x in payload if isinstance(x, dict)]
        if isinstance(payload, dict):
            rows = payload.get("data", [])
            if isinstance(rows, list):
                return [x for x in rows if isinstance(x, dict)]
        raise ValueError(f"Unsupported JSON structure in {path}")

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parsed = json.loads(line)
            if isinstance(parsed, dict):
                rows.append(parsed)
    return rows


def load_records(data_path: str) -> List[Dict[str, Any]]:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    rows = _parse_json_or_jsonl(path)
    logger.info("Loaded %d raw rows from %s", len(rows), path)
    return rows


def format_training_text(rows: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    formatted: List[Dict[str, str]] = []
    for row in rows:
        text = ""
        if isinstance(row.get("text"), str) and row["text"].strip():
            text = row["text"].strip()
        elif isinstance(row.get("prompt"), str) and isinstance(row.get("completion"), str):
            text = f"{row['prompt']}{row['completion']}".strip()

        if text:
            formatted.append({"text": text})
    if not formatted:
        raise ValueError("No usable training rows found (expected text or prompt+completion fields).")
    logger.info("Prepared %d formatted training rows", len(formatted))
    return formatted


def _env_requests_tpu() -> bool:
    if os.environ.get("PJRT_DEVICE", "").strip().upper() == "TPU":
        return True

    kaggle_accel = os.environ.get("KAGGLE_ACCELERATOR_TYPE", "").strip().upper()
    if kaggle_accel.startswith("TPU"):
        return True

    tpu_hint_keys = ("TPU_NAME", "COLAB_TPU_ADDR", "TPU_WORKER_ID")
    return any(bool(os.environ.get(k)) for k in tpu_hint_keys)


def tpu_available() -> bool:
    try:
        import torch_xla.core.xla_model as xm  # type: ignore

        device = str(xm.xla_device()).lower()
        return "xla" in device
    except Exception:
        # If environment hints TPU but torch_xla is missing/broken, treat as unavailable.
        return False


def detect_accelerator(requested: str) -> str:
    req = (requested or "auto").strip().lower()
    valid = {"auto", "cuda", "tpu", "cpu"}
    if req not in valid:
        raise ValueError(f"--accelerator must be one of {sorted(valid)}, got {requested!r}")

    if req != "auto":
        return req

    if _env_requests_tpu():
        return "tpu"

    if tpu_available():
        return "tpu"

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def latest_checkpoint(output_dir: str) -> str | None:
    out = Path(output_dir)
    if not out.exists():
        return None
    checkpoints = sorted(out.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    return str(checkpoints[-1]) if checkpoints else None


def resolve_target_modules(model: Any, requested: List[str]) -> List[str]:
    module_names = [name for name, _ in model.named_modules()]
    name_set = set(module_names)

    requested_present = [target for target in requested if any(name.endswith(target) for name in name_set)]
    if requested_present:
        return requested_present

    fallback_groups = [
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ["c_attn", "c_proj", "c_fc"],
        ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    ]
    for group in fallback_groups:
        present = [target for target in group if any(name.endswith(target) for name in name_set)]
        if present:
            logger.warning("Requested target modules not found; using fallback targets: %s", present)
            return present

    raise ValueError(
        "Could not find LoRA target modules in the base model. "
        "Please update TrainingConfig.target_modules for this architecture."
    )


def run_training(config: TrainingConfig, data_path: str) -> str:
    # Work around environments with incompatible torchao/triton versions.
    # QLoRA/LoRA training in this script does not depend on torchao.
    try:
        import importlib.util as iu2

        original_find_spec = iu2.find_spec

        def patched_find_spec(name: str, *args: Any, **kwargs: Any):
            if name == "torchao":
                return None
            return original_find_spec(name, *args, **kwargs)

        iu2.find_spec = patched_find_spec  # type: ignore[assignment]
    except Exception:
        pass

    try:
        import transformers.utils.import_utils as iu  # type: ignore

        iu._torchao_available = False
        iu._torchao_version = "0.0.0"
    except Exception:
        pass

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise ImportError(
            f"Missing required package: {exc}. Install transformers/peft/datasets/accelerate first."
        ) from exc

    accelerator = detect_accelerator(config.accelerator)
    config.accelerator = accelerator
    if accelerator == "tpu":
        os.environ.setdefault("PJRT_DEVICE", "TPU")
        if not tpu_available():
            raise RuntimeError(
                "TPU accelerator requested but torch_xla TPU device is unavailable. "
                "Use a TPU runtime and ensure torch_xla matches your torch version."
            )

    if accelerator != "cuda" and config.use_4bit:
        logger.warning("4-bit QLoRA is CUDA-only. Disabling use_4bit for accelerator=%s.", accelerator)
        config.use_4bit = False

    logger.info("Selected accelerator: %s", accelerator)
    if accelerator == "cuda" and torch.cuda.is_available():
        logger.info("CUDA device: %s", torch.cuda.get_device_name(0))
        logger.info("GPU memory: %.1f GB", torch.cuda.get_device_properties(0).total_memory / 1e9)
    elif accelerator == "tpu":
        logger.info("TPU mode enabled (PyTorch/XLA).")

    records = load_records(data_path)
    formatted = format_training_text(records)
    dataset = Dataset.from_list(formatted)

    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def tokenize_batch(batch: Dict[str, List[str]]) -> Dict[str, Any]:
        tokenized = tokenizer(
            batch["text"],
            truncation=True,
            max_length=config.max_seq_length,
            padding="max_length",
        )
        labels: List[List[int]] = []
        for ids, mask in zip(tokenized["input_ids"], tokenized["attention_mask"]):
            labels.append([token_id if attn == 1 else -100 for token_id, attn in zip(ids, mask)])
        tokenized["labels"] = labels
        return tokenized

    tokenized_dataset = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing",
    )

    torch_dtype = torch.float32
    if accelerator == "tpu":
        torch_dtype = torch.bfloat16
    elif accelerator == "cuda":
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model_load_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
    }
    if config.use_4bit:
        compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype, torch.bfloat16)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=config.use_nested_quant,
        )
        model_load_kwargs["quantization_config"] = bnb_config
        model_load_kwargs["device_map"] = "auto"
    else:
        model_load_kwargs["torch_dtype"] = torch_dtype

    logger.info("Loading base model: %s", config.base_model)
    model = AutoModelForCausalLM.from_pretrained(config.base_model, **model_load_kwargs)
    model.config.use_cache = False

    if config.use_4bit:
        model = prepare_model_for_kbit_training(model)
    elif config.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    resolved_targets = resolve_target_modules(model, config.target_modules)
    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=resolved_targets,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    use_bf16 = accelerator == "tpu" or (accelerator == "cuda" and torch.cuda.is_bf16_supported())
    use_fp16 = accelerator == "cuda" and not use_bf16
    optimizer_name = "paged_adamw_32bit" if (accelerator == "cuda" and config.use_4bit) else "adamw_torch"

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        bf16=use_bf16,
        fp16=use_fp16,
        optim=optimizer_name,
        dataloader_pin_memory=(accelerator == "cuda"),
        remove_unused_columns=False,
        gradient_checkpointing=config.gradient_checkpointing,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    resume_ckpt = latest_checkpoint(config.output_dir)
    if resume_ckpt:
        logger.info("Resuming from checkpoint: %s", resume_ckpt)

    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=resume_ckpt)

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    with (Path(config.output_dir) / "training_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, indent=2)

    logger.info("Training complete. Adapter/model saved to: %s", config.output_dir)
    return config.output_dir


def run_dry_run(config: TrainingConfig, data_path: str) -> None:
    rows = load_records(data_path)
    formatted = format_training_text(rows)
    accelerator = detect_accelerator(config.accelerator)
    if accelerator == "tpu" and not tpu_available():
        logger.warning(
            "Dry-run selected TPU but torch_xla TPU device is not available in this runtime."
        )
    use_4bit = config.use_4bit and accelerator == "cuda"
    logger.info("=== DRY RUN ===")
    logger.info("rows=%d formatted=%d", len(rows), len(formatted))
    logger.info("accelerator=%s use_4bit=%s", accelerator, use_4bit)
    logger.info("base_model=%s output_dir=%s", config.base_model, config.output_dir)
    logger.info("epochs=%d batch_size=%d grad_acc=%d", config.num_epochs, config.per_device_train_batch_size, config.gradient_accumulation_steps)
    logger.info("max_seq_length=%d learning_rate=%g", config.max_seq_length, config.learning_rate)
    logger.info("=== DRY RUN COMPLETE ===")


def main() -> None:
    parser = argparse.ArgumentParser(description="BD-NSCA LoRA/QLoRA training")
    parser.add_argument("--data", required=True, help="Path to training data (.json or .jsonl)")
    parser.add_argument("--output-dir", default="outputs/adapter", help="Output directory")
    parser.add_argument("--base-model", default="microsoft/Phi-3-mini-4k-instruct", help="Base model")
    parser.add_argument("--accelerator", default="auto", help="auto|cuda|tpu|cpu")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--use-4bit", dest="use_4bit", action="store_true")
    parser.add_argument("--no-4bit", dest="use_4bit", action="store_false")
    parser.set_defaults(use_4bit=True)
    parser.add_argument("--kaggle-autopath", dest="kaggle_autopath", action="store_true")
    parser.add_argument("--no-kaggle-autopath", dest="kaggle_autopath", action="store_false")
    parser.set_defaults(kaggle_autopath=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    resolved_data = args.data
    resolved_output_dir = args.output_dir
    if args.kaggle_autopath:
        resolved_data, resolved_output_dir = resolve_kaggle_paths(
            data_path=str(args.data),
            output_dir=str(args.output_dir),
        )

    config = TrainingConfig(
        base_model=args.base_model,
        output_dir=resolved_output_dir,
        accelerator=args.accelerator,
        use_4bit=args.use_4bit,
        num_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        max_seq_length=args.max_seq_length,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    if args.dry_run:
        run_dry_run(config, resolved_data)
        return
    run_training(config, resolved_data)


if __name__ == "__main__":
    main()

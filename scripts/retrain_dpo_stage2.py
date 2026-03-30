#!/usr/bin/env python3
"""Retrain Stage-2 DPO on mined + synthetic conflict-state preference pairs."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import logging
import os
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SEED_DEFAULT = 42

SYSTEM_TEMPLATE = (
    "You are {persona}, stationed at {location}. "
    "Current state: {behavior_state}. "
    "Game context: {game_state}. "
    "Respond in character."
)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def format_prompt_from_row(row: Dict[str, Any]) -> str:
    prompt = row.get("prompt", "")
    if isinstance(prompt, str):
        text = prompt.strip()
        if text:
            return text
        return "<|system|>\nRespond in character.\n<|assistant|>\n"

    if isinstance(prompt, dict):
        prompt_text = str(prompt.get("prompt_text", "")).strip()
        if prompt_text:
            return prompt_text
        system = SYSTEM_TEMPLATE.format(
            persona=str(prompt.get("persona", "")).strip(),
            location=str(prompt.get("location", "")).strip(),
            behavior_state=str(prompt.get("behavior_state", "")).strip(),
            game_state=str(prompt.get("game_state", "{}")).strip(),
        )
        player_input = str(prompt.get("player_input", "")).strip()
        return f"<|system|>\n{system}\n<|user|>\n{player_input}\n<|assistant|>\n"

    return "<|system|>\nRespond in character.\n<|assistant|>\n"


def extract_behavior_state(row: Dict[str, Any]) -> str:
    direct = str(row.get("behavior_state", "")).strip()
    if direct:
        return direct
    prompt = row.get("prompt", {})
    if isinstance(prompt, dict):
        return str(prompt.get("behavior_state", "")).strip() or "unknown"
    return "unknown"


def stratified_split(rows: List[Dict[str, Any]], eval_frac: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_state: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        state = extract_behavior_state(row)
        by_state[state].append(row)

    rng = random.Random(seed)
    train_rows: List[Dict[str, Any]] = []
    eval_rows: List[Dict[str, Any]] = []
    for state, items in by_state.items():
        copy = list(items)
        rng.shuffle(copy)
        n_eval = max(1, int(len(copy) * eval_frac)) if len(copy) > 1 else 0
        eval_rows.extend(copy[:n_eval])
        train_rows.extend(copy[n_eval:])

    rng.shuffle(train_rows)
    rng.shuffle(eval_rows)
    return train_rows, eval_rows


def build_dataset(rows: List[Dict[str, Any]]):
    datasets_mod = importlib.import_module("datasets")
    Dataset = getattr(datasets_mod, "Dataset")

    prompts: List[str] = []
    chosens: List[str] = []
    rejecteds: List[str] = []
    states: List[str] = []
    for row in rows:
        chosen = str(row.get("chosen", "")).strip()
        rejected = str(row.get("rejected", "")).strip()
        if not chosen or not rejected or chosen == rejected:
            continue
        prompts.append(format_prompt_from_row(row))
        chosens.append(chosen)
        rejecteds.append(rejected)
        states.append(extract_behavior_state(row))
    return Dataset.from_dict(
        {
            "prompt": prompts,
            "chosen": chosens,
            "rejected": rejecteds,
            "behavior_state": states,
        }
    )


@dataclass
class Stage2Config:
    pairs: str
    base_model: str
    output_dir: str
    eval_frac: float = 0.1
    beta: float = 0.1
    lr: float = 5e-6
    epochs: int = 2
    batch_size: int = 1
    grad_accum: int = 8
    max_prompt_length: int = 512
    max_length: int = 1024
    use_4bit: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    seed: int = SEED_DEFAULT


def resolve_target_modules(model: Any, requested: List[str]) -> List[str]:
    module_names = [name for name, _ in model.named_modules()]
    present = [t for t in requested if any(name.endswith(t) for name in module_names)]
    if present:
        return present
    fallback = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    present_fb = [t for t in fallback if any(name.endswith(t) for name in module_names)]
    if present_fb:
        return present_fb
    raise RuntimeError("Could not resolve LoRA target modules for base model.")


def detect_accelerator() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def state_distribution(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        s = extract_behavior_state(row)
        counts[s] = int(counts.get(s, 0)) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-2 DPO retraining with conflict-state targeting.")
    parser.add_argument("--pairs", required=True, help="JSONL preference pairs from pipeline2_dpo_mining.py")
    parser.add_argument("--base-model", required=True, help="Path to existing SFT checkpoint")
    parser.add_argument("--out-dir", default="storage/outputs/checkpoints/dpo_v2")
    parser.add_argument("--eval-frac", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    args = parser.parse_args()

    cfg = Stage2Config(
        pairs=str(args.pairs),
        base_model=str(args.base_model),
        output_dir=str(args.out_dir),
        eval_frac=float(args.eval_frac),
        beta=float(args.beta),
        lr=float(args.lr),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        grad_accum=int(args.grad_accum),
        max_prompt_length=int(args.max_prompt_length),
        max_length=int(args.max_length),
        use_4bit=not bool(args.no_4bit),
        lora_r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        seed=int(args.seed),
    )

    pairs_path = Path(cfg.pairs)
    if not pairs_path.exists():
        raise FileNotFoundError(f"Pairs file not found: {pairs_path}")
    all_rows = read_jsonl(pairs_path)
    if not all_rows:
        raise ValueError(f"No rows in pairs file: {pairs_path}")

    train_rows, eval_rows = stratified_split(all_rows, eval_frac=max(0.0, min(0.5, cfg.eval_frac)), seed=cfg.seed)
    if not train_rows:
        raise RuntimeError("No training rows left after stratified split.")

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "train_pairs.jsonl", train_rows)
    write_jsonl(out_dir / "eval_pairs.jsonl", eval_rows)

    train_ds = build_dataset(train_rows)
    eval_ds = build_dataset(eval_rows) if eval_rows else None
    logger.info("Rows: total=%d train=%d eval=%d", len(all_rows), len(train_ds), len(eval_ds) if eval_ds is not None else 0)

    accel = detect_accelerator()
    logger.info("Accelerator: %s", accel)

    import torch
    peft_mod = importlib.import_module("peft")
    LoraConfig = getattr(peft_mod, "LoraConfig")
    get_peft_model = getattr(peft_mod, "get_peft_model")
    prepare_model_for_kbit_training = getattr(peft_mod, "prepare_model_for_kbit_training")
    transformers_mod = importlib.import_module("transformers")
    AutoModelForCausalLM = getattr(transformers_mod, "AutoModelForCausalLM")
    AutoTokenizer = getattr(transformers_mod, "AutoTokenizer")
    BitsAndBytesConfig = getattr(transformers_mod, "BitsAndBytesConfig")
    TrainingArguments = getattr(transformers_mod, "TrainingArguments")

    try:
        trl_dpo_mod = importlib.import_module("trl.trainer.dpo_trainer")
        DPOTrainer = getattr(trl_dpo_mod, "DPOTrainer")
    except Exception as exc:
        raise RuntimeError("Missing TRL DPOTrainer. Install/upgrade `trl`.") from exc
    try:
        trl_cfg_mod = importlib.import_module("trl.trainer.dpo_config")
        TRLDPOConfig = getattr(trl_cfg_mod, "DPOConfig")
    except Exception:
        TRLDPOConfig = None

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

    base_args: Dict[str, Any] = dict(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy=("epoch" if eval_ds is not None else "no"),
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
        "ref_model": None,
        "args": train_args,
        "train_dataset": train_ds,
    }
    if eval_ds is not None:
        trainer_kwargs["eval_dataset"] = eval_ds

    sig = inspect.signature(DPOTrainer.__init__).parameters
    if "tokenizer" in sig:
        trainer_kwargs["tokenizer"] = tokenizer
    if "processing_class" in sig:
        trainer_kwargs["processing_class"] = tokenizer
    if "beta" in sig:
        trainer_kwargs["beta"] = cfg.beta
    if "max_prompt_length" in sig:
        trainer_kwargs["max_prompt_length"] = cfg.max_prompt_length
    if "max_length" in sig:
        trainer_kwargs["max_length"] = cfg.max_length

    trainer = DPOTrainer(**trainer_kwargs)
    trainer.train()

    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    manifest = {
        "pairs_path": str(pairs_path),
        "base_model": cfg.base_model,
        "output_dir": cfg.output_dir,
        "counts": {"total": len(all_rows), "train": len(train_ds), "eval": len(eval_ds) if eval_ds is not None else 0},
        "state_distribution": {
            "all": state_distribution(all_rows),
            "train": state_distribution(train_rows),
            "eval": state_distribution(eval_rows),
        },
        "hyperparameters": {
            "beta": cfg.beta,
            "lr": cfg.lr,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "grad_accum": cfg.grad_accum,
            "max_prompt_length": cfg.max_prompt_length,
            "max_length": cfg.max_length,
            "use_4bit": cfg.use_4bit,
            "lora_r": cfg.lora_r,
            "lora_alpha": cfg.lora_alpha,
            "lora_dropout": cfg.lora_dropout,
            "target_modules": targets,
            "seed": cfg.seed,
        },
        "accelerator": accel,
    }
    write_json(out_dir / "run_manifest.json", manifest)
    logger.info("Saved checkpoint: %s", cfg.output_dir)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()

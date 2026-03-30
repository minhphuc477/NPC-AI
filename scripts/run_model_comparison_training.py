#!/usr/bin/env python3
"""Run the same SFT->DPO pipeline across multiple base models.

Default model set is compute-matched for 4GB-class VRAM:
- microsoft/Phi-3-mini-4k-instruct
- microsoft/Phi-3.5-mini-instruct
- google/gemma-2-2b-it
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


MODEL_ALIASES: Dict[str, str] = {
    "phi3-mini": "microsoft/Phi-3-mini-4k-instruct",
    "phi3.5-mini": "microsoft/Phi-3.5-mini-instruct",
    "gemma2-2b": "google/gemma-2-2b-it",
    "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
    "llama3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
}

MODEL_PROFILES: Dict[str, List[str]] = {
    "none": [],
    "minimal_4gb": ["phi3-mini", "gemma2-2b"],
    "laptop_safe": ["phi3-mini", "phi3.5-mini", "gemma2-2b", "qwen2.5-3b"],
    "extended_4gb": ["phi3-mini", "phi3.5-mini", "gemma2-2b", "qwen2.5-3b", "llama3.2-3b"],
}


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_models(raw: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for token in [x.strip() for x in str(raw).split(",") if x.strip()]:
        model_id = MODEL_ALIASES.get(token.lower(), token)
        if model_id in seen:
            continue
        seen.add(model_id)
        out.append(model_id)
    if not out:
        raise ValueError("No models selected. Provide --models with at least one entry.")
    return out


def resolve_models(raw: str, profile: str) -> List[str]:
    merged: List[str] = []
    merged.extend([x.strip() for x in str(raw).split(",") if x.strip()])
    prof = str(profile or "none").strip().lower()
    merged.extend(MODEL_PROFILES.get(prof, []))
    return parse_models(",".join(merged))


def short_name(model_id: str) -> str:
    base = model_id.split("/")[-1].strip().lower()
    clean = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in base)
    while "--" in clean:
        clean = clean.replace("--", "-")
    return clean.strip("-") or "model"


def run_command(command: List[str], *, env: Dict[str, str] | None = None, dry_run: bool = False) -> None:
    cmd_str = " ".join(shlex.quote(x) for x in command)
    print(f"[cmd] {cmd_str}")
    if dry_run:
        return
    proc = subprocess.run(command, check=False, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {cmd_str}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


@dataclass
class ComparisonConfig:
    models: List[str]
    sft_data: str
    dpo_dataset: str
    output_root: str
    accelerator: str
    use_4bit: bool
    seed: int

    # SFT
    sft_epochs: int
    sft_max_steps: int
    sft_batch_size: int
    sft_grad_acc: int
    sft_lr: float
    sft_max_seq_length: int
    sft_lora_r: int
    sft_lora_alpha: int
    sft_lora_dropout: float
    sft_max_grad_norm: float
    sft_gradient_checkpointing: bool

    # DPO
    dpo_epochs: int
    dpo_max_steps: int
    dpo_batch_size: int
    dpo_grad_acc: int
    dpo_lr: float
    dpo_beta: float
    dpo_max_prompt_length: int
    dpo_max_length: int
    dpo_lora_r: int
    dpo_lora_alpha: int
    dpo_lora_dropout: float
    dpo_max_grad_norm: float

    # Orchestration
    merge_sft_for_dpo: bool
    force_cpu_merge: bool
    skip_sft: bool
    skip_dpo: bool
    dry_run: bool


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run fair model comparison with shared SFT->DPO configs."
    )
    parser.add_argument(
        "--models",
        default="phi3-mini,phi3.5-mini,gemma2-2b",
        help="Comma list of aliases/model IDs. Aliases: phi3-mini, phi3.5-mini, gemma2-2b, qwen2.5-3b, llama3.2-3b",
    )
    parser.add_argument(
        "--model-profile",
        default="none",
        choices=sorted(MODEL_PROFILES.keys()),
        help="Optional profile to expand --models with compute-matched model sets.",
    )
    parser.add_argument("--sft-data", required=True, help="SFT training data (.json/.jsonl)")
    parser.add_argument("--dpo-dataset", required=True, help="DPO preference dataset (.json/.jsonl)")
    parser.add_argument("--output-root", default="storage/runs/model_comparison")
    parser.add_argument("--accelerator", default="auto", choices=["auto", "cuda", "cpu", "tpu"])
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--skip-sft", action="store_true")
    parser.add_argument("--skip-dpo", action="store_true")
    parser.add_argument("--no-merge-sft-for-dpo", dest="merge_sft_for_dpo", action="store_false")
    parser.add_argument("--force-cpu-merge", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.set_defaults(merge_sft_for_dpo=True)

    # SFT defaults (4GB-safe baseline)
    parser.add_argument("--sft-epochs", type=int, default=2)
    parser.add_argument("--sft-max-steps", type=int, default=-1)
    parser.add_argument("--sft-batch-size", type=int, default=1)
    parser.add_argument("--sft-grad-acc", type=int, default=8)
    parser.add_argument("--sft-lr", type=float, default=2e-4)
    parser.add_argument("--sft-max-seq-length", type=int, default=768)
    parser.add_argument("--sft-lora-r", type=int, default=16)
    parser.add_argument("--sft-lora-alpha", type=int, default=32)
    parser.add_argument("--sft-lora-dropout", type=float, default=0.05)
    parser.add_argument("--sft-max-grad-norm", type=float, default=1.0)
    parser.add_argument("--disable-sft-gradient-checkpointing", action="store_true")

    # DPO defaults (4GB-safe baseline)
    parser.add_argument("--dpo-epochs", type=int, default=2)
    parser.add_argument("--dpo-max-steps", type=int, default=-1)
    parser.add_argument("--dpo-batch-size", type=int, default=1)
    parser.add_argument("--dpo-grad-acc", type=int, default=8)
    parser.add_argument("--dpo-lr", type=float, default=5e-6)
    parser.add_argument("--dpo-beta", type=float, default=0.1)
    parser.add_argument("--dpo-max-prompt-length", type=int, default=512)
    parser.add_argument("--dpo-max-length", type=int, default=1024)
    parser.add_argument("--dpo-lora-r", type=int, default=16)
    parser.add_argument("--dpo-lora-alpha", type=int, default=32)
    parser.add_argument("--dpo-lora-dropout", type=float, default=0.05)
    parser.add_argument("--dpo-max-grad-norm", type=float, default=1.0)

    args = parser.parse_args()

    sft_data = Path(args.sft_data)
    dpo_dataset = Path(args.dpo_dataset)
    if not sft_data.exists():
        raise FileNotFoundError(f"SFT data not found: {sft_data}")
    if not dpo_dataset.exists():
        raise FileNotFoundError(f"DPO dataset not found: {dpo_dataset}")

    cfg = ComparisonConfig(
        models=resolve_models(args.models, args.model_profile),
        sft_data=str(sft_data),
        dpo_dataset=str(dpo_dataset),
        output_root=str(args.output_root),
        accelerator=str(args.accelerator),
        use_4bit=not bool(args.no_4bit),
        seed=int(args.seed),
        sft_epochs=int(args.sft_epochs),
        sft_max_steps=int(args.sft_max_steps),
        sft_batch_size=int(args.sft_batch_size),
        sft_grad_acc=int(args.sft_grad_acc),
        sft_lr=float(args.sft_lr),
        sft_max_seq_length=int(args.sft_max_seq_length),
        sft_lora_r=int(args.sft_lora_r),
        sft_lora_alpha=int(args.sft_lora_alpha),
        sft_lora_dropout=float(args.sft_lora_dropout),
        sft_max_grad_norm=float(args.sft_max_grad_norm),
        sft_gradient_checkpointing=not bool(args.disable_sft_gradient_checkpointing),
        dpo_epochs=int(args.dpo_epochs),
        dpo_max_steps=int(args.dpo_max_steps),
        dpo_batch_size=int(args.dpo_batch_size),
        dpo_grad_acc=int(args.dpo_grad_acc),
        dpo_lr=float(args.dpo_lr),
        dpo_beta=float(args.dpo_beta),
        dpo_max_prompt_length=int(args.dpo_max_prompt_length),
        dpo_max_length=int(args.dpo_max_length),
        dpo_lora_r=int(args.dpo_lora_r),
        dpo_lora_alpha=int(args.dpo_lora_alpha),
        dpo_lora_dropout=float(args.dpo_lora_dropout),
        dpo_max_grad_norm=float(args.dpo_max_grad_norm),
        merge_sft_for_dpo=bool(args.merge_sft_for_dpo),
        force_cpu_merge=bool(args.force_cpu_merge),
        skip_sft=bool(args.skip_sft),
        skip_dpo=bool(args.skip_dpo),
        dry_run=bool(args.dry_run),
    )

    run_root = Path(cfg.output_root) / utc_stamp()
    run_root.mkdir(parents=True, exist_ok=True)
    write_json(run_root / "comparison_config.json", asdict(cfg))

    results: List[Dict[str, Any]] = []
    for model_id in cfg.models:
        name = short_name(model_id)
        model_root = run_root / name
        sft_adapter = model_root / "sft_adapter"
        sft_merged = model_root / "sft_merged"
        dpo_adapter = model_root / "dpo_adapter"
        model_root.mkdir(parents=True, exist_ok=True)

        row: Dict[str, Any] = {
            "model_id": model_id,
            "short_name": name,
            "paths": {
                "model_root": str(model_root),
                "sft_adapter": str(sft_adapter),
                "sft_merged": str(sft_merged),
                "dpo_adapter": str(dpo_adapter),
            },
            "status": "pending",
        }
        try:
            if not cfg.skip_sft:
                sft_cmd = [
                    sys.executable,
                    "scripts/train_qlora.py",
                    "--data",
                    cfg.sft_data,
                    "--output-dir",
                    str(sft_adapter),
                    "--base-model",
                    model_id,
                    "--accelerator",
                    cfg.accelerator,
                    "--epochs",
                    str(cfg.sft_epochs),
                    "--max-steps",
                    str(cfg.sft_max_steps),
                    "--batch-size",
                    str(cfg.sft_batch_size),
                    "--gradient-accumulation-steps",
                    str(cfg.sft_grad_acc),
                    "--learning-rate",
                    str(cfg.sft_lr),
                    "--max-seq-length",
                    str(cfg.sft_max_seq_length),
                    "--lora-r",
                    str(cfg.sft_lora_r),
                    "--lora-alpha",
                    str(cfg.sft_lora_alpha),
                    "--lora-dropout",
                    str(cfg.sft_lora_dropout),
                    "--max-grad-norm",
                    str(cfg.sft_max_grad_norm),
                ]
                if cfg.sft_gradient_checkpointing:
                    sft_cmd.append("--gradient-checkpointing")
                if cfg.use_4bit:
                    sft_cmd.append("--use-4bit")
                else:
                    sft_cmd.append("--no-4bit")
                run_command(sft_cmd, dry_run=cfg.dry_run)

            dpo_base = model_id
            if cfg.merge_sft_for_dpo and not cfg.skip_dpo:
                merge_cmd = [
                    sys.executable,
                    "scripts/merge_lora.py",
                    "--base-model",
                    model_id,
                    "--lora-path",
                    str(sft_adapter),
                    "--output-path",
                    str(sft_merged),
                ]
                merge_env: Dict[str, str] | None = None
                if cfg.force_cpu_merge:
                    merge_env = dict(os.environ)
                    merge_env["CUDA_VISIBLE_DEVICES"] = ""
                run_command(merge_cmd, env=merge_env, dry_run=cfg.dry_run)
                dpo_base = str(sft_merged)

            if not cfg.skip_dpo:
                dpo_cmd = [
                    sys.executable,
                    "scripts/train_dpo.py",
                    "--dataset",
                    cfg.dpo_dataset,
                    "--base-model",
                    dpo_base,
                    "--output-dir",
                    str(dpo_adapter),
                    "--accelerator",
                    cfg.accelerator,
                    "--epochs",
                    str(cfg.dpo_epochs),
                    "--max-steps",
                    str(cfg.dpo_max_steps),
                    "--batch-size",
                    str(cfg.dpo_batch_size),
                    "--grad-acc",
                    str(cfg.dpo_grad_acc),
                    "--lr",
                    str(cfg.dpo_lr),
                    "--beta",
                    str(cfg.dpo_beta),
                    "--max-prompt-length",
                    str(cfg.dpo_max_prompt_length),
                    "--max-length",
                    str(cfg.dpo_max_length),
                    "--lora-r",
                    str(cfg.dpo_lora_r),
                    "--lora-alpha",
                    str(cfg.dpo_lora_alpha),
                    "--lora-dropout",
                    str(cfg.dpo_lora_dropout),
                    "--max-grad-norm",
                    str(cfg.dpo_max_grad_norm),
                    "--seed",
                    str(cfg.seed),
                ]
                if not cfg.use_4bit:
                    dpo_cmd.append("--no-4bit")
                run_command(dpo_cmd, dry_run=cfg.dry_run)

            row["status"] = "ok"
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = str(exc)
            results.append(row)
            write_json(run_root / "comparison_results.json", results)
            raise

        results.append(row)
        write_json(run_root / "comparison_results.json", results)

    print(f"Comparison run root: {run_root}")
    print(f"Results: {run_root / 'comparison_results.json'}")


if __name__ == "__main__":
    main()

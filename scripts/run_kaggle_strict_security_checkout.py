#!/usr/bin/env python3
"""Run strict full checkout with required security benchmark checks."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], dry_run: bool) -> None:
    printable = " ".join(cmd)
    print(f"[run] {printable}")
    if dry_run:
        return
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {printable}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict security-required full checkout profile.")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--candidate-model", default="elara-npc:latest")
    parser.add_argument("--baseline-model", default="phi3:mini")
    parser.add_argument("--baseline-models", default="phi3:latest")
    parser.add_argument("--scenario-file", default="data/proposal_eval_scenarios_large_v2.jsonl")
    parser.add_argument("--batch-size", type=int, default=28)
    parser.add_argument("--proposal-max-tokens", type=int, default=80)
    parser.add_argument("--proposal-temperature", type=float, default=0.2)
    parser.add_argument("--proposal-min-arm-success-rate", type=float, default=0.90)
    parser.add_argument("--gate-min-external-significant-wins", type=int, default=10)
    parser.add_argument("--gate-min-human-soft-win-rate", type=float, default=0.50)
    parser.add_argument("--gate-min-human-kappa", type=float, default=0.20)
    parser.add_argument("--serving-max-tokens", type=int, default=64)
    parser.add_argument("--serving-temperature", type=float, default=0.2)
    parser.add_argument("--output-root", default="artifacts/final_checkout_strict")
    parser.add_argument("--proposal-run", default="")
    parser.add_argument("--publication-run", default="")
    parser.add_argument("--skip-ablation-baselines", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cmd = [
        sys.executable,
        str(Path("scripts") / "run_kaggle_full_results.py"),
        "--host",
        str(args.host),
        "--candidate-model",
        str(args.candidate_model),
        "--baseline-model",
        str(args.baseline_model),
        "--baseline-models",
        str(args.baseline_models),
        "--scenario-file",
        str(args.scenario_file),
        "--batch-size",
        str(args.batch_size),
        "--proposal-max-tokens",
        str(args.proposal_max_tokens),
        "--proposal-temperature",
        str(args.proposal_temperature),
        "--proposal-min-arm-success-rate",
        str(args.proposal_min_arm_success_rate),
        "--serving-max-tokens",
        str(args.serving_max_tokens),
        "--serving-temperature",
        str(args.serving_temperature),
        "--gate-min-external-significant-wins",
        str(args.gate_min_external_significant_wins),
        "--gate-min-human-soft-win-rate",
        str(args.gate_min_human_soft_win_rate),
        "--gate-min-human-kappa",
        str(args.gate_min_human_kappa),
        "--output-root",
        str(args.output_root),
        "--run-security-benchmark",
        "--require-security-benchmark",
    ]

    if args.proposal_run:
        cmd.extend(["--proposal-run", str(args.proposal_run)])
    if args.publication_run:
        cmd.extend(["--publication-run", str(args.publication_run)])
    if args.skip_ablation_baselines:
        cmd.append("--skip-ablation-baselines")
    if args.dry_run:
        cmd.append("--dry-run")

    run_command(cmd, dry_run=bool(args.dry_run))


if __name__ == "__main__":
    main()

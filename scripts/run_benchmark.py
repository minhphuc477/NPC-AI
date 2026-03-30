#!/usr/bin/env python3
"""Thin compatibility wrapper for the publication benchmark suite.

Legacy `run_benchmark.py` used a fragile interactive subprocess flow and could
report false passes. This wrapper delegates to the maintained, reproducible
pipeline in `run_publication_benchmark_suite.py`.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run publication benchmark suite (compat wrapper).")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--candidate-model", default="elara-npc:latest")
    parser.add_argument("--baseline-model", default="phi3:mini")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=56)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--run-security-benchmark", action="store_true")
    parser.add_argument("--run-security-spoofed-benchmark", action="store_true")
    parser.add_argument("--skip-ablation-baselines", action="store_true")
    parser.add_argument("--output-root", default="")
    args = parser.parse_args()

    command = [
        sys.executable,
        str(Path("scripts") / "run_publication_benchmark_suite.py"),
        "--host",
        str(args.host),
        "--candidate-model",
        str(args.candidate_model),
        "--baseline-model",
        str(args.baseline_model),
        "--repeats",
        str(int(args.repeats)),
        "--max-tokens",
        str(int(args.max_tokens)),
        "--temperature",
        str(float(args.temperature)),
    ]
    if args.run_security_benchmark:
        command.append("--run-security-benchmark")
    if args.run_security_spoofed_benchmark:
        command.append("--run-security-spoofed-benchmark")
    if args.skip_ablation_baselines:
        command.append("--skip-ablation-baselines")
    if str(args.output_root).strip():
        command.extend(["--output-root", str(args.output_root).strip()])

    result = subprocess.run(command, check=False)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()

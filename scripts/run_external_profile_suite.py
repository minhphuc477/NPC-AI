#!/usr/bin/env python3
"""Run publication benchmark suite across multiple benchmark profiles."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def parse_profiles(raw: str) -> List[str]:
    out: List[str] = []
    for token in [x.strip() for x in str(raw).split(",") if x.strip()]:
        if token not in out:
            out.append(token)
    return out


def profile_args(profile: str) -> Dict[str, str]:
    p = profile.strip().lower()
    if p == "core":
        return {
            "prompts": "data/serving_prompts.jsonl",
            "serving_references": "data/serving_references.jsonl",
            "retrieval_corpus": "data/retrieval_corpus.jsonl",
            "retrieval_gold": "data/retrieval_gold.jsonl",
            "security_dataset": "data/retrieval_poison_benchmark.jsonl",
        }
    if p == "wide":
        return {
            "prompts": "data/serving_prompts_wide.jsonl",
            "serving_references": "data/serving_references_wide.jsonl",
            "retrieval_corpus": "data/retrieval_corpus_wide.jsonl",
            "retrieval_gold": "data/retrieval_gold_wide.jsonl",
            "security_dataset": "data/retrieval_poison_benchmark_large.jsonl",
        }
    raise ValueError(f"Unknown profile: {profile}")


def latest_subdir(root: Path) -> Path:
    dirs = [p for p in root.iterdir() if p.is_dir()]
    if not dirs:
        raise FileNotFoundError(f"No run dirs under: {root}")
    return sorted(dirs, key=lambda p: p.name)[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run publication benchmark suite on multiple profiles.")
    parser.add_argument("--profiles", default="core,wide")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--candidate-model", default="elara-npc:latest")
    parser.add_argument("--baseline-model", default="phi3:mini")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=59)
    parser.add_argument("--run-security-benchmark", action="store_true")
    parser.add_argument("--run-security-spoofed-benchmark", action="store_true")
    parser.add_argument("--security-benchmark-exe", default="cpp/build/Release/bench_retrieval_security.exe")
    parser.add_argument("--output-root", default="artifacts/publication_profiles")
    args = parser.parse_args()

    profiles = parse_profiles(args.profiles)
    if not profiles:
        raise RuntimeError("No profiles selected.")

    suite_root = Path(args.output_root) / utc_stamp()
    suite_root.mkdir(parents=True, exist_ok=True)

    runs: List[Dict[str, Any]] = []
    for idx, profile in enumerate(profiles):
        cfg = profile_args(profile)
        profile_out = suite_root / profile
        profile_out.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            "scripts/run_publication_benchmark_suite.py",
            "--host",
            str(args.host),
            "--candidate-model",
            str(args.candidate_model),
            "--baseline-model",
            str(args.baseline_model),
            "--prompts",
            cfg["prompts"],
            "--serving-references",
            cfg["serving_references"],
            "--retrieval-corpus",
            cfg["retrieval_corpus"],
            "--retrieval-gold",
            cfg["retrieval_gold"],
            "--security-dataset",
            cfg["security_dataset"],
            "--repeats",
            str(args.repeats),
            "--max-tokens",
            str(args.max_tokens),
            "--temperature",
            str(args.temperature),
            "--seed",
            str(int(args.seed) + idx * 13),
            "--security-benchmark-exe",
            str(args.security_benchmark_exe),
            "--output-root",
            str(profile_out),
        ]
        if args.run_security_benchmark:
            cmd.append("--run-security-benchmark")
        if args.run_security_spoofed_benchmark:
            cmd.append("--run-security-spoofed-benchmark")

        print(f"[profile] {profile}: running publication suite")
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"Profile run failed ({profile}) with code {proc.returncode}")

        run_dir = latest_subdir(profile_out)
        runs.append({"profile": profile, "run_dir": str(run_dir), "returncode": proc.returncode})

    manifest = {
        "suite_root": str(suite_root),
        "profiles": profiles,
        "runs": runs,
    }
    write_json(suite_root / "manifest.json", manifest)
    print(f"Profile suite completed: {suite_root}")
    print(f"Manifest: {suite_root / 'manifest.json'}")


if __name__ == "__main__":
    main()


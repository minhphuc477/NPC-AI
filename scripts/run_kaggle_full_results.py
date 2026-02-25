#!/usr/bin/env python3
"""Run the full proposal/publication result pipeline and emit a single manifest."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import requests


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def latest_subdir(root: Path) -> Path:
    dirs = [p for p in root.iterdir() if p.is_dir()]
    if not dirs:
        raise FileNotFoundError(f"No run directories found under: {root}")
    return sorted(dirs, key=lambda p: p.name)[-1]


def resolve_run_path(raw: str, root: Path) -> Path:
    token = str(raw or "").strip()
    if not token:
        raise ValueError("empty run token")
    if token.lower() == "latest":
        return latest_subdir(root)
    path = Path(token)
    if path.exists():
        return path
    candidate = root / token
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Cannot resolve run path '{raw}' under {root}")


def run_command(command: List[str], dry_run: bool = False) -> None:
    printable = " ".join(command)
    print(f"[run] {printable}")
    if dry_run:
        return
    proc = subprocess.run(command, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {printable}")


def ollama_ready(host: str, model_names: List[str], timeout_s: int = 10) -> None:
    try:
        resp = requests.get(f"{host}/api/tags", timeout=timeout_s)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        raise RuntimeError(f"Ollama host is unavailable at {host}: {exc}") from exc
    models = {
        str(item.get("name", "")).strip()
        for item in payload.get("models", [])
        if str(item.get("name", "")).strip()
    }
    missing = [m for m in model_names if m and m not in models]
    if missing:
        raise RuntimeError(f"Missing Ollama models at {host}: {missing}")


def maybe_generate_inputs(dry_run: bool) -> None:
    if not Path("data/retrieval_gold_wide.jsonl").exists() or not Path("data/retrieval_corpus_wide.jsonl").exists():
        run_command(
            [
                sys.executable,
                "scripts/generate_wide_benchmark_sets.py",
                "--docs-per-domain",
                "24",
                "--queries-per-domain",
                "20",
                "--prompts-per-domain",
                "4",
            ],
            dry_run=dry_run,
        )

    if not Path("data/retrieval_poison_benchmark_large.jsonl").exists():
        run_command(
            [
                sys.executable,
                "scripts/generate_poison_benchmark_large.py",
                "--target-size",
                "400",
            ],
            dry_run=dry_run,
        )

    if not Path("data/proposal_eval_scenarios_large.jsonl").exists():
        run_command(
            [
                sys.executable,
                "scripts/generate_proposal_scenarios_large.py",
                "--variants-per-base",
                "14",
                "--output",
                "data/proposal_eval_scenarios_large.jsonl",
            ],
            dry_run=dry_run,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full result pipeline for Kaggle/repro checkout.")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--candidate-model", default="elara-npc:latest")
    parser.add_argument("--baseline-model", default="phi3:mini")
    parser.add_argument("--baseline-models", default="phi3:latest")
    parser.add_argument("--scenario-file", default="data/proposal_eval_scenarios_large.jsonl")
    parser.add_argument(
        "--proposal-run",
        default="",
        help="Reuse an existing proposal run (path/run_id/latest) and skip proposal generation.",
    )
    parser.add_argument(
        "--publication-run",
        default="",
        help="Reuse an existing publication run (path/run_id/latest) and skip publication benchmark generation.",
    )
    parser.add_argument("--batch-size", type=int, default=28)
    parser.add_argument("--proposal-max-tokens", type=int, default=80)
    parser.add_argument("--proposal-temperature", type=float, default=0.2)
    parser.add_argument("--multirater-annotators", default="phi3:mini|balanced|0.00,phi3:mini|balanced|0.05,phi3:mini|balanced|0.10")
    parser.add_argument("--multirater-scenarios", type=int, default=36)
    parser.add_argument("--serving-models", default="elara-npc:latest,phi3:mini,phi3:latest")
    parser.add_argument("--serving-max-tokens", type=int, default=64)
    parser.add_argument("--serving-temperature", type=float, default=0.2)
    parser.add_argument(
        "--skip-ablation-baselines",
        action="store_true",
        help="Skip keyword/random retrieval ablation baselines in publication suite.",
    )
    parser.add_argument("--run-security-benchmark", action="store_true")
    parser.add_argument("--require-security-benchmark", action="store_true")
    parser.add_argument("--output-root", default="artifacts/final_checkout")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-missing-ollama",
        action="store_true",
        help="If Ollama host/models are unavailable, continue in dry-run mode instead of failing.",
    )
    args = parser.parse_args()

    if bool(args.require_security_benchmark) and not bool(args.run_security_benchmark):
        raise ValueError("--require-security-benchmark requires --run-security-benchmark")

    if not args.dry_run:
        model_checks = [str(args.candidate_model), str(args.baseline_model)]
        for token in str(args.baseline_models).split(","):
            t = token.strip()
            if t:
                model_checks.append(t)
        try:
            ollama_ready(args.host, model_checks)
        except RuntimeError as exc:
            if not args.allow_missing_ollama:
                raise
            print(f"[warn] {exc}")
            print("[warn] Falling back to dry-run mode (--allow-missing-ollama).")
            args.dry_run = True

    maybe_generate_inputs(dry_run=bool(args.dry_run))

    proposal_token = str(args.proposal_run).strip()
    if proposal_token:
        proposal_run = resolve_run_path(proposal_token, Path("artifacts/proposal"))
    else:
        # 1) Proposal batched eval.
        run_command(
            [
                sys.executable,
                "scripts/run_proposal_alignment_eval_batched.py",
                "--host",
                str(args.host),
                "--candidate-model",
                str(args.candidate_model),
                "--baseline-model",
                str(args.baseline_model),
                "--baseline-models",
                str(args.baseline_models),
                "--scenarios",
                str(args.scenario_file),
                "--batch-size",
                str(args.batch_size),
                "--repeats",
                "1",
                "--max-tokens",
                str(args.proposal_max_tokens),
                "--temperature",
                str(args.proposal_temperature),
                "--bertscore-model-type",
                "roberta-large",
                "--bertscore-batch-size",
                "16",
            ],
            dry_run=bool(args.dry_run),
        )
        proposal_run = Path("artifacts/proposal/latest")
        if not args.dry_run:
            proposal_run = latest_subdir(Path("artifacts/proposal"))

    # 2) Multi-rater campaign + attach.
    human_csv = proposal_run / "human_eval_llm_multirater_consistent.csv"
    run_command(
        [
            sys.executable,
            "scripts/run_llm_multirater_campaign.py",
            "--run-dir",
            str(proposal_run),
            "--host",
            str(args.host),
            "--annotators",
            str(args.multirater_annotators),
            "--arms",
            "proposed_contextual_controlled,baseline_no_context,baseline_no_context_phi3_latest",
            "--scenario-limit",
            str(args.multirater_scenarios),
            "--max-tokens",
            "160",
            "--timeout-s",
            "120",
            "--output-csv",
            str(human_csv),
        ],
        dry_run=bool(args.dry_run),
    )
    run_command(
        [
            sys.executable,
            "scripts/attach_human_eval_to_run.py",
            "--run-dir",
            str(proposal_run),
            "--human-eval-file",
            str(human_csv),
        ],
        dry_run=bool(args.dry_run),
    )

    # 3) Lexical benchmark.
    run_command(
        [
            sys.executable,
            "scripts/run_lexical_diversity_benchmark.py",
            "--run-dir",
            str(proposal_run),
        ],
        dry_run=bool(args.dry_run),
    )

    publication_token = str(args.publication_run).strip()
    if publication_token:
        publication_run = resolve_run_path(publication_token, Path("artifacts/publication"))
    else:
        # 4) Publication suite (single canonical run for quality gate).
        pub_cmd = [
            sys.executable,
            "scripts/run_publication_benchmark_suite.py",
            "--host",
            str(args.host),
            "--candidate-model",
            str(args.candidate_model),
            "--baseline-model",
            str(args.baseline_model),
            "--repeats",
            "1",
            "--max-tokens",
            str(args.serving_max_tokens),
            "--temperature",
            str(args.serving_temperature),
            "--reranker-pairs",
            "data/retrieval_reranker_pairs_wide.jsonl",
            "--reranker-train-frac",
            "0.85",
            "--reranker-bm25-candidate-k",
            "24",
        ]
        if args.run_security_benchmark:
            pub_cmd.extend(["--run-security-benchmark", "--run-security-spoofed-benchmark"])
        if args.skip_ablation_baselines:
            pub_cmd.append("--skip-ablation-baselines")
        run_command(pub_cmd, dry_run=bool(args.dry_run))

        publication_run = Path("artifacts/publication/latest")
        if not args.dry_run:
            publication_run = latest_subdir(Path("artifacts/publication"))

    # 5) Serving matrix.
    run_command(
        [
            sys.executable,
            "scripts/run_serving_efficiency_matrix.py",
            "--host",
            str(args.host),
            "--models",
            str(args.serving_models),
            "--candidate-model",
            str(args.candidate_model),
            "--prompts",
            "data/serving_prompts_wide.jsonl",
            "--serving-references",
            "data/serving_references_wide.jsonl",
            "--max-tokens",
            str(args.serving_max_tokens),
            "--temperature",
            str(args.serving_temperature),
            "--repeats",
            "1",
        ],
        dry_run=bool(args.dry_run),
    )
    serving_run = Path("artifacts/serving_efficiency/latest")
    if not args.dry_run:
        serving_run = latest_subdir(Path("artifacts/serving_efficiency"))

    # 6) External profile suite (core + wide).
    run_command(
        [
            sys.executable,
            "scripts/run_external_profile_suite.py",
            "--profiles",
            "core,wide",
            "--host",
            str(args.host),
            "--candidate-model",
            str(args.candidate_model),
            "--baseline-model",
            str(args.baseline_model),
            "--repeats",
            "1",
            "--max-tokens",
            str(args.serving_max_tokens),
            "--temperature",
            str(args.serving_temperature),
        ],
        dry_run=bool(args.dry_run),
    )
    profile_suite = Path("artifacts/publication_profiles/latest")
    if not args.dry_run:
        profile_suite = latest_subdir(Path("artifacts/publication_profiles"))

    # 7) Significant-improvement data builders.
    run_command(
        [
            sys.executable,
            "scripts/build_preference_dataset_from_eval.py",
            "--run-dir",
            str(proposal_run),
            "--human-eval-file",
            str(human_csv),
            "--target-arm",
            "proposed_contextual_controlled",
            "--baseline-arms",
            "baseline_no_context,baseline_no_context_phi3_latest",
            "--metric",
            "overall_quality",
            "--min-raters",
            "2",
            "--min-margin",
            "0.25",
        ],
        dry_run=bool(args.dry_run),
    )
    run_command(
        [
            sys.executable,
            "scripts/build_retrieval_hard_negative_set.py",
            "--retrieval-gold",
            "data/retrieval_gold_wide.jsonl",
            "--retrieval-corpus",
            "data/retrieval_corpus_wide.jsonl",
            "--hard-negatives-per-query",
            "10",
            "--cross-domain-negatives-per-query",
            "4",
        ],
        dry_run=bool(args.dry_run),
    )

    # 8) Quality gate.
    gate_cmd = [
        sys.executable,
        "scripts/proposal_quality_gate.py",
        "--proposal-run",
        str(proposal_run),
        "--publication-run",
        str(publication_run),
        "--require-human-eval",
        "--output-json",
        str(proposal_run / "quality_gate_report_final.json"),
        "--output-md",
        str(proposal_run / "quality_gate_report_final.md"),
    ]
    if args.require_security_benchmark:
        gate_cmd.append("--require-security-benchmark")
    run_command(gate_cmd, dry_run=bool(args.dry_run))

    manifest_root = Path(args.output_root) / utc_stamp()
    manifest = {
        "host": str(args.host),
        "candidate_model": str(args.candidate_model),
        "baseline_model": str(args.baseline_model),
        "proposal_run": str(proposal_run),
        "publication_run": str(publication_run),
        "serving_efficiency_run": str(serving_run),
        "publication_profile_suite": str(profile_suite),
        "skip_ablation_baselines": bool(args.skip_ablation_baselines),
        "human_eval_csv": str(human_csv),
        "quality_gate_json": str(proposal_run / "quality_gate_report_final.json"),
        "quality_gate_md": str(proposal_run / "quality_gate_report_final.md"),
        "preference_dataset": str(proposal_run / "preference_dataset.jsonl"),
        "retrieval_hard_negatives": "data/retrieval_hard_negatives_wide.jsonl",
        "retrieval_reranker_pairs": "data/retrieval_reranker_pairs_wide.jsonl",
        "dry_run": bool(args.dry_run),
    }
    write_json(manifest_root / "manifest.json", manifest)
    print(f"Final checkout manifest: {manifest_root / 'manifest.json'}")


if __name__ == "__main__":
    main()

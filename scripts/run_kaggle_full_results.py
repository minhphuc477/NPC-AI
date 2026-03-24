#!/usr/bin/env python3
"""Run the full proposal/publication result pipeline and emit a single manifest."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List

import requests
try:
    from local_model_profiles import (
        baseline_profile_choices,
        fetch_ollama_models,
        format_model_csv,
        resolve_baseline_models,
        split_available_missing,
    )
except ModuleNotFoundError:
    from scripts.local_model_profiles import (
        baseline_profile_choices,
        fetch_ollama_models,
        format_model_csv,
        resolve_baseline_models,
        split_available_missing,
    )

STORAGE_ARTIFACT_ROOT = Path("storage/artifacts")
LEGACY_ARTIFACT_ROOT = Path("artifacts")


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def latest_subdir(root: Path) -> Path:
    dirs = [
        p
        for p in root.iterdir()
        if p.is_dir() and p.name != "latest" and not p.name.endswith("_batch_tmp")
    ]
    if not dirs:
        raise FileNotFoundError(f"No run directories found under: {root}")
    return sorted(dirs, key=lambda p: (p.stat().st_mtime, p.name))[-1]


def resolve_run_path(raw: str, root: Path, fallback_root: Path | None = None) -> Path:
    token = str(raw or "").strip()
    if not token:
        raise ValueError("empty run token")
    if token.lower() == "latest":
        try:
            return latest_subdir(root)
        except FileNotFoundError:
            if fallback_root is not None:
                return latest_subdir(fallback_root)
            raise
    path = Path(token)
    if path.exists():
        return path
    candidate = root / token
    if candidate.exists():
        return candidate
    if fallback_root is not None:
        fallback = fallback_root / token
        if fallback.exists():
            return fallback
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
    if not Path("data/retrieval_gold_wide_v2.jsonl").exists() or not Path("data/retrieval_corpus_wide_v2.jsonl").exists():
        run_command(
            [
                sys.executable,
                "scripts/generate_wide_benchmark_sets.py",
                "--docs-per-domain", "28",
                "--queries-per-domain", "24",
                "--prompts-per-domain", "6",
                "--retrieval-corpus-out", "data/retrieval_corpus_wide_v2.jsonl",
                "--retrieval-gold-out", "data/retrieval_gold_wide_v2.jsonl",
                "--serving-prompts-out", "data/serving_prompts_wide_v2.jsonl",
                "--serving-references-out", "data/serving_references_wide_v2.jsonl",
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

    if not Path("data/proposal_eval_scenarios_large_v2.jsonl").exists():
        run_command(
            [
                sys.executable,
                "scripts/generate_proposal_scenarios_large.py",
                "--variants-per-base",
                "18",
                "--max-player-jaccard", "0.72",
                "--rephrase-attempts", "16",
                "--min-template-signature-ratio", "0.35",
                "--fail-on-low-diversity",
                "--output",
                "data/proposal_eval_scenarios_large_v2.jsonl",
            ],
            dry_run=dry_run,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full result pipeline for Kaggle/repro checkout.")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--candidate-model", default="elara-npc:latest")
    parser.add_argument("--baseline-model", default="phi3:mini")
    parser.add_argument("--baseline-models", default="phi3:latest")
    parser.add_argument(
        "--baseline-profile",
        default="none",
        choices=baseline_profile_choices(),
        help="Optional baseline profile to expand baseline-models with laptop-safe packs.",
    )
    parser.add_argument("--scenario-file", default="data/proposal_eval_scenarios_large_v2.jsonl")
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
    parser.add_argument(
        "--proposal-control-alt-profile",
        choices=[
            "none",
            "runtime_optimized",
            "quality_optimized",
            "risk_latency_aware",
            "hybrid_balanced",
            "intent_focus_adaptive",
            "blend_balanced",
            "custom",
        ],
        default="none",
        help="Optional alternate response-control profile to add as a second controlled arm.",
    )
    parser.add_argument(
        "--proposal-control-alt-arm-id",
        default="proposed_contextual_controlled_alt",
        help="Arm ID for alternate proposal control profile.",
    )
    parser.add_argument(
        "--proposal-control-alt-overrides-file",
        default="",
        help="Optional JSON overrides file for custom alternate control profile.",
    )
    parser.add_argument(
        "--proposal-target-arm",
        default="proposed_contextual_controlled",
        help="Target arm used for external-baseline, human-eval, and quality-gate comparisons.",
    )
    parser.add_argument(
        "--proposal-min-arm-success-rate",
        type=float,
        default=0.90,
        help="Minimum required successful-request rate per arm in proposal eval.",
    )
    parser.add_argument(
        "--multirater-annotators",
        default="phi3:mini|balanced|0.00,phi3:mini|balanced|0.05,phi3:mini|balanced|0.10",
    )
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
    parser.add_argument(
        "--gate-min-external-significant-wins",
        type=int,
        default=8,
        help="Minimum significantly-positive metrics per external baseline for proposal gate.",
    )
    parser.add_argument(
        "--gate-min-human-soft-win-rate",
        type=float,
        default=0.50,
        help="Minimum human soft-win rate vs each baseline for proposal gate.",
    )
    parser.add_argument(
        "--gate-min-human-kappa",
        type=float,
        default=0.20,
        help="Minimum mean pairwise kappa for the human-eval gate.",
    )
    parser.add_argument("--output-root", default="storage/artifacts/final_checkout")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-missing-ollama",
        action="store_true",
        help="If Ollama host/models are unavailable, continue in dry-run mode instead of failing.",
    )
    parser.add_argument(
        "--allow-missing-baselines",
        action="store_true",
        help="Prune missing baseline-models; candidate-model and baseline-model remain required.",
    )
    args = parser.parse_args()
    resolved_baselines = resolve_baseline_models(str(args.baseline_models), str(args.baseline_profile))
    if not resolved_baselines:
        resolved_baselines = [str(args.baseline_model)]
    args.baseline_models = format_model_csv(resolved_baselines)

    proposal_root = STORAGE_ARTIFACT_ROOT / "proposal"
    proposal_fallback_root = LEGACY_ARTIFACT_ROOT / "proposal"
    publication_root = STORAGE_ARTIFACT_ROOT / "publication"
    publication_fallback_root = LEGACY_ARTIFACT_ROOT / "publication"
    serving_root = STORAGE_ARTIFACT_ROOT / "serving_efficiency"
    serving_fallback_root = LEGACY_ARTIFACT_ROOT / "serving_efficiency"
    profiles_root = STORAGE_ARTIFACT_ROOT / "publication_profiles"
    profiles_fallback_root = LEGACY_ARTIFACT_ROOT / "publication_profiles"

    if bool(args.require_security_benchmark) and not bool(args.run_security_benchmark):
        raise ValueError("--require-security-benchmark requires --run-security-benchmark")

    if not args.dry_run:
        required_models = [str(args.candidate_model), str(args.baseline_model)]
        baseline_tokens = [t.strip() for t in str(args.baseline_models).split(",") if t.strip()]
        model_checks = required_models + baseline_tokens
        try:
            ollama_ready(args.host, model_checks)
        except RuntimeError as exc:
            if bool(args.allow_missing_baselines):
                try:
                    installed = fetch_ollama_models(str(args.host), timeout_s=10)
                    required_avail, required_missing = split_available_missing(required_models, installed)
                    if required_missing:
                        raise RuntimeError(
                            f"Missing required models (cannot prune): {required_missing}"
                        ) from exc
                    baseline_avail, baseline_missing = split_available_missing(baseline_tokens, installed)
                    if not baseline_avail:
                        baseline_avail = [str(args.baseline_model)]
                    args.baseline_models = format_model_csv(baseline_avail)
                    model_checks = required_avail + baseline_avail
                    ollama_ready(args.host, model_checks)
                    print(
                        f"[info] Pruned missing baselines: {baseline_missing}. "
                        f"Using baseline-models={args.baseline_models}"
                    )
                except RuntimeError as inner_exc:
                    exc = inner_exc
                except Exception as fallback_exc:
                    exc = RuntimeError(f"Unexpected baseline-pruning error: {fallback_exc}")
                else:
                    exc = None
            if exc is not None and not args.allow_missing_ollama:
                raise exc
            elif exc is not None:
                print(f"[warn] {exc}")
                print("[warn] Falling back to dry-run mode (--allow-missing-ollama).")
                args.dry_run = True

    maybe_generate_inputs(dry_run=bool(args.dry_run))

    proposal_token = str(args.proposal_run).strip()
    target_arm = str(args.proposal_target_arm).strip() or "proposed_contextual_controlled"
    multirater_arms = [target_arm, "baseline_no_context", "baseline_no_context_phi3_latest"]

    if proposal_token:
        proposal_run = resolve_run_path(proposal_token, proposal_root, proposal_fallback_root)
    else:
        # 1) Proposal batched eval.
        proposal_cmd = [
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
            "--min-arm-success-rate",
            str(args.proposal_min_arm_success_rate),
            "--bertscore-model-type",
            "roberta-large",
            "--bertscore-batch-size",
            "16",
            "--target-arm",
            target_arm,
            "--control-alt-profile",
            str(args.proposal_control_alt_profile),
            "--control-alt-arm-id",
            str(args.proposal_control_alt_arm_id),
        ]
        overrides_file = str(args.proposal_control_alt_overrides_file).strip()
        if overrides_file:
            proposal_cmd.extend(["--control-alt-overrides-file", overrides_file])
        run_command(proposal_cmd, dry_run=bool(args.dry_run))
        proposal_run = proposal_root / "latest"
        if not args.dry_run:
            if proposal_root.exists():
                proposal_run = latest_subdir(proposal_root)
            else:
                proposal_run = latest_subdir(proposal_fallback_root)

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
            ",".join(multirater_arms),
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
        publication_run = resolve_run_path(publication_token, publication_root, publication_fallback_root)
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
            "--prompts",
            "data/serving_prompts_wide_v2.jsonl",
            "--serving-references",
            "data/serving_references_wide_v2.jsonl",
            "--retrieval-corpus",
            "data/retrieval_corpus_wide_v2.jsonl",
            "--retrieval-gold",
            "data/retrieval_gold_wide_v2.jsonl",
            "--reranker-pairs",
            "data/retrieval_reranker_pairs_wide.jsonl",
            "--reranker-train-frac",
            "0.85",
            "--reranker-bm25-candidate-k",
            "24",
            "--security-dataset",
            "data/retrieval_poison_benchmark_large.jsonl",
        ]
        if args.run_security_benchmark:
            pub_cmd.extend(["--run-security-benchmark", "--run-security-spoofed-benchmark"])
        if args.skip_ablation_baselines:
            pub_cmd.append("--skip-ablation-baselines")
        run_command(pub_cmd, dry_run=bool(args.dry_run))

        publication_run = publication_root / "latest"
        if not args.dry_run:
            if publication_root.exists():
                publication_run = latest_subdir(publication_root)
            else:
                publication_run = latest_subdir(publication_fallback_root)

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
            "data/serving_prompts_wide_v2.jsonl",
            "--serving-references",
            "data/serving_references_wide_v2.jsonl",
            "--max-tokens",
            str(args.serving_max_tokens),
            "--temperature",
            str(args.serving_temperature),
            "--repeats",
            "1",
        ],
        dry_run=bool(args.dry_run),
    )
    serving_run = serving_root / "latest"
    if not args.dry_run:
        if serving_root.exists():
            serving_run = latest_subdir(serving_root)
        else:
            serving_run = latest_subdir(serving_fallback_root)

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
    profile_suite = profiles_root / "latest"
    if not args.dry_run:
        if profiles_root.exists():
            profile_suite = latest_subdir(profiles_root)
        else:
            profile_suite = latest_subdir(profiles_fallback_root)

    # 6b) Aggregate runtime evidence from proposal-style interactive runs.
    runtime_json = profile_suite / "live_runtime_summary.json"
    runtime_md = profile_suite / "live_runtime_summary.md"
    run_command(
        [
            sys.executable,
            "scripts/analyze_live_runtime_conditions.py",
            "--root",
            str(proposal_root if proposal_root.exists() or not proposal_fallback_root.exists() else proposal_fallback_root),
            "--arm",
            target_arm,
            "--output-json",
            str(runtime_json),
            "--output-md",
            str(runtime_md),
        ],
        dry_run=bool(args.dry_run),
    )

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
            target_arm,
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
            "data/retrieval_gold_wide_v2.jsonl",
            "--retrieval-corpus",
            "data/retrieval_corpus_wide_v2.jsonl",
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
        "--min-external-significant-wins",
        str(args.gate_min_external_significant_wins),
        "--min-human-soft-win-rate",
        str(args.gate_min_human_soft_win_rate),
        "--min-human-kappa",
        str(args.gate_min_human_kappa),
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
        "live_runtime_summary_json": str(runtime_json),
        "live_runtime_summary_md": str(runtime_md),
        "skip_ablation_baselines": bool(args.skip_ablation_baselines),
        "proposal_target_arm": target_arm,
        "proposal_control_alt_profile": str(args.proposal_control_alt_profile),
        "proposal_control_alt_arm_id": str(args.proposal_control_alt_arm_id),
        "proposal_control_alt_overrides_file": str(args.proposal_control_alt_overrides_file),
        "human_eval_csv": str(human_csv),
        "multirater_annotators": str(args.multirater_annotators),
        "gate_min_external_significant_wins": int(args.gate_min_external_significant_wins),
        "gate_min_human_soft_win_rate": float(args.gate_min_human_soft_win_rate),
        "gate_min_human_kappa": float(args.gate_min_human_kappa),
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

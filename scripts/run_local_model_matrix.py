#!/usr/bin/env python3
"""Run inference-only model matrix across local Ollama tags on a fixed scenario set."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

try:
    from local_model_profiles import (
        BASELINE_PROFILES,
        fetch_ollama_models,
        format_model_csv,
        merge_unique_models,
        parse_model_csv,
        split_available_missing,
    )
except ModuleNotFoundError:
    from scripts.local_model_profiles import (
        BASELINE_PROFILES,
        fetch_ollama_models,
        format_model_csv,
        merge_unique_models,
        parse_model_csv,
        split_available_missing,
    )


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def latest_run(root: Path) -> Path:
    dirs = [p for p in root.iterdir() if p.is_dir() and not p.name.endswith("_batch_tmp")]
    if not dirs:
        raise FileNotFoundError(f"No run found under: {root}")
    return sorted(dirs, key=lambda p: p.name)[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--profile", default="laptop_safe", choices=sorted(BASELINE_PROFILES.keys()))
    parser.add_argument("--models", default="", help="Optional extra model tags.")
    parser.add_argument("--scenarios", default="data/proposal_eval_scenarios_large.jsonl")
    parser.add_argument("--max-scenarios", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=80)
    parser.add_argument("--output-root", default="storage/artifacts/model_matrix")
    parser.add_argument("--allow-missing-models", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    requested = merge_unique_models(BASELINE_PROFILES.get(str(args.profile), []), parse_model_csv(str(args.models)))
    if not requested:
        requested = ["phi3:mini"]

    if not bool(args.dry_run):
        installed = fetch_ollama_models(str(args.host), timeout_s=12)
        available, missing = split_available_missing(requested, installed)
        if missing and not bool(args.allow_missing_models):
            raise RuntimeError(f"Missing models: {missing}")
    else:
        available = requested
        missing = []

    run_root = Path(args.output_root) / utc_stamp()
    run_root.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []

    for model_tag in available:
        model_id = model_tag.replace(":", "_").replace("/", "_")
        model_out = run_root / model_id / "proposal_runs"
        cmd = [
            sys.executable,
            "scripts/run_proposal_alignment_eval.py",
            "--host",
            str(args.host),
            "--candidate-model",
            model_tag,
            "--baseline-model",
            model_tag,
            "--skip-external-baselines",
            "--scenarios",
            str(args.scenarios),
            "--max-scenarios",
            str(int(args.max_scenarios)),
            "--repeats",
            str(int(args.repeats)),
            "--temperature",
            str(float(args.temperature)),
            "--max-tokens",
            str(int(args.max_tokens)),
            "--output-root",
            str(model_out),
        ]
        if bool(args.dry_run):
            print("[dry-run]", " ".join(cmd))
            results.append({"model": model_tag, "status": "dry_run"})
            continue
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            results.append(
                {
                    "model": model_tag,
                    "status": "failed",
                    "error": (proc.stderr or proc.stdout or "").strip()[-2000:],
                }
            )
            continue

        latest = latest_run(model_out)
        summary = json.loads((latest / "summary.json").read_text(encoding="utf-8"))
        ops = json.loads((latest / "operational_metrics.json").read_text(encoding="utf-8"))
        arm = summary.get("proposed_contextual_controlled", {}) or summary.get("proposed_contextual", {})
        op_arm = ops.get("arms", {}).get("proposed_contextual_controlled", {}) or ops.get("arms", {}).get(
            "proposed_contextual", {}
        )
        row = {
            "model": model_tag,
            "status": "ok",
            "run_dir": str(latest),
            "overall_quality_mean": float(arm.get("overall_quality", {}).get("mean", float("nan"))),
            "context_relevance_mean": float(arm.get("context_relevance", {}).get("mean", float("nan"))),
            "persona_consistency_mean": float(arm.get("persona_consistency", {}).get("mean", float("nan"))),
            "fallback_rate": float(op_arm.get("fallback_rate", float("nan"))),
            "retry_rate": float(op_arm.get("retry_rate", float("nan"))),
            "latency_p95_ms": float(op_arm.get("latency_ms", {}).get("p95", float("nan"))),
        }
        results.append(row)

    summary_out = {
        "profile": str(args.profile),
        "requested_models": requested,
        "available_models": available,
        "missing_models": missing,
        "rows": results,
    }
    write_json(run_root / "model_matrix_summary.json", summary_out)

    lines: List[str] = ["# Local Model Matrix", ""]
    lines.append(f"- Profile: `{args.profile}`")
    lines.append(f"- Baselines: `{format_model_csv(available)}`")
    lines.append("")
    lines.append("| Model | Status | Overall | Context | Persona | Fallback | Retry | Latency p95 (ms) |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in results:
        lines.append(
            f"| {r.get('model')} | {r.get('status')} | "
            f"{float(r.get('overall_quality_mean', float('nan'))):.4f} | "
            f"{float(r.get('context_relevance_mean', float('nan'))):.4f} | "
            f"{float(r.get('persona_consistency_mean', float('nan'))):.4f} | "
            f"{float(r.get('fallback_rate', float('nan'))):.4f} | "
            f"{float(r.get('retry_rate', float('nan'))):.4f} | "
            f"{float(r.get('latency_p95_ms', float('nan'))):.2f} |"
        )
    (run_root / "model_matrix_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"saved_summary={run_root / 'model_matrix_summary.json'}")


if __name__ == "__main__":
    main()

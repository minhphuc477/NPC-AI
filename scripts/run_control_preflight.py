#!/usr/bin/env python3
"""Run multi-seed preflight checks for response-control stability before full 112-scenario runs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Sequence

try:
    from local_model_profiles import baseline_profile_choices, format_model_csv, resolve_baseline_models
except ModuleNotFoundError:
    from scripts.local_model_profiles import baseline_profile_choices, format_model_csv, resolve_baseline_models


SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_SCRIPT = SCRIPT_DIR / "run_proposal_alignment_eval.py"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_seeds(raw: str) -> List[int]:
    out: List[int] = []
    for chunk in raw.replace(";", ",").split(","):
        token = chunk.strip()
        if not token:
            continue
        out.append(int(token))
    dedup: List[int] = []
    seen = set()
    for seed in out:
        if seed in seen:
            continue
        seen.add(seed)
        dedup.append(seed)
    if not dedup:
        raise ValueError("No seeds provided.")
    return dedup


def latest_subdir(root: Path) -> Path:
    dirs = [p for p in root.iterdir() if p.is_dir()]
    if not dirs:
        raise FileNotFoundError(f"No run directory found under: {root}")
    return sorted(dirs, key=lambda p: p.name)[-1]


def max_pairwise_abs(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    diffs = [abs(a - b) for a, b in combinations(values, 2)]
    return max(diffs) if diffs else 0.0


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def paired_key_for_arm(arm_id: str) -> str:
    arm = str(arm_id or "").strip()
    if arm == "proposed_contextual_controlled":
        return "controlled_vs_proposed_raw"
    return "controlled_alt_vs_proposed_raw"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run two-seed (or multi-seed) control preflight.")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--candidate-model", default="elara-npc:latest")
    parser.add_argument("--baseline-model", default="phi3:mini")
    parser.add_argument("--baseline-models", default="phi3:mini")
    parser.add_argument(
        "--baseline-profile",
        default="none",
        choices=baseline_profile_choices(),
        help="Optional baseline profile to expand baseline-models with laptop-safe packs.",
    )
    parser.add_argument("--scenarios", default="data/proposal_eval_scenarios_112_diverse.jsonl")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-scenarios", type=int, default=40)
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument("--min-arm-success-rate", type=float, default=0.90)
    parser.add_argument("--bertscore-lang", default="en")
    parser.add_argument("--bertscore-model-type", default="")
    parser.add_argument("--bertscore-batch-size", type=int, default=16)
    parser.add_argument("--bertscore-cache-dir", default="")
    parser.add_argument("--control-adaptive-low-rewrites", type=int, default=3)
    parser.add_argument(
        "--control-alt-profile",
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
    )
    parser.add_argument("--enable-control-intent-risk-adaptation", action="store_true")
    parser.add_argument("--enable-control-latency-adaptation", action="store_true")
    parser.add_argument("--enable-control-intent-focused-context", action="store_true")
    parser.add_argument("--control-latency-relax-start-pressure", type=float, default=0.55)
    parser.add_argument("--control-latency-relax-max-delta", type=float, default=0.12)
    parser.add_argument("--control-intent-focus-min-keep", type=int, default=3)
    parser.add_argument("--control-intent-focus-keep-ratio-low", type=float, default=0.45)
    parser.add_argument("--control-intent-focus-keep-ratio-medium", type=float, default=0.65)
    parser.add_argument("--control-intent-focus-keep-ratio-high", type=float, default=1.0)
    parser.add_argument("--control-intent-focus-min-relevance", type=float, default=0.20)
    parser.add_argument("--enable-control-near-pass", action="store_true")
    parser.add_argument("--control-near-pass-max-context-gap", type=float, default=0.05)
    parser.add_argument("--control-near-pass-max-persona-gap", type=float, default=0.04)
    parser.add_argument("--control-near-pass-score-floor", type=float, default=0.34)
    parser.add_argument("--disable-control-near-pass-block-high-risk", action="store_true")
    parser.add_argument("--control-alt-arm-id", default="proposed_contextual_controlled_alt")
    parser.add_argument("--seeds", default="29,31")
    parser.add_argument("--gate-arm", default="proposed_contextual_controlled")
    parser.add_argument("--max-fallback-rate", type=float, default=0.45)
    parser.add_argument("--max-retry-rate", type=float, default=0.75)
    parser.add_argument("--min-first-pass-accept-rate", type=float, default=0.25)
    parser.add_argument("--max-overall-drift", type=float, default=0.05)
    parser.add_argument("--max-context-drift", type=float, default=0.06)
    parser.add_argument("--max-persona-drift", type=float, default=0.12)
    parser.add_argument("--max-naturalness-drift", type=float, default=0.06)
    parser.add_argument("--max-fallback-drift", type=float, default=0.15)
    parser.add_argument("--max-retry-drift", type=float, default=0.15)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--output-root", default="storage/artifacts/proposal_control_tuning/preflight")
    args = parser.parse_args()
    resolved_baselines = resolve_baseline_models(str(args.baseline_models), str(args.baseline_profile))
    if not resolved_baselines:
        resolved_baselines = [str(args.baseline_model)]
    args.baseline_models = format_model_csv(resolved_baselines)

    seeds = parse_seeds(str(args.seeds))
    gate_arm = str(args.gate_arm).strip() or "proposed_contextual_controlled"
    paired_key = paired_key_for_arm(gate_arm)
    preflight_dir = Path(args.output_root) / utc_stamp()
    seed_runs_root = preflight_dir / "seed_runs"
    seed_runs_root.mkdir(parents=True, exist_ok=True)

    per_seed: List[Dict[str, Any]] = []
    for seed in seeds:
        seed_out = seed_runs_root / f"seed_{seed}"
        seed_out.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(EVAL_SCRIPT),
            "--host",
            str(args.host),
            "--candidate-model",
            str(args.candidate_model),
            "--baseline-model",
            str(args.baseline_model),
            "--baseline-models",
            str(args.baseline_models),
            "--scenarios",
            str(args.scenarios),
            "--temperature",
            str(args.temperature),
            "--max-tokens",
            str(args.max_tokens),
            "--repeats",
            str(args.repeats),
            "--max-scenarios",
            str(args.max_scenarios),
            "--seed",
            str(seed),
            "--timeout-s",
            str(args.timeout_s),
            "--min-arm-success-rate",
            str(args.min_arm_success_rate),
            "--bertscore-lang",
            str(args.bertscore_lang),
            "--bertscore-model-type",
            str(args.bertscore_model_type),
            "--bertscore-batch-size",
            str(args.bertscore_batch_size),
            "--bertscore-cache-dir",
            str(args.bertscore_cache_dir),
            "--control-adaptive-low-rewrites",
            str(args.control_adaptive_low_rewrites),
            "--control-latency-relax-start-pressure",
            str(args.control_latency_relax_start_pressure),
            "--control-latency-relax-max-delta",
            str(args.control_latency_relax_max_delta),
            "--control-intent-focus-min-keep",
            str(args.control_intent_focus_min_keep),
            "--control-intent-focus-keep-ratio-low",
            str(args.control_intent_focus_keep_ratio_low),
            "--control-intent-focus-keep-ratio-medium",
            str(args.control_intent_focus_keep_ratio_medium),
            "--control-intent-focus-keep-ratio-high",
            str(args.control_intent_focus_keep_ratio_high),
            "--control-intent-focus-min-relevance",
            str(args.control_intent_focus_min_relevance),
            "--control-near-pass-max-context-gap",
            str(args.control_near_pass_max_context_gap),
            "--control-near-pass-max-persona-gap",
            str(args.control_near_pass_max_persona_gap),
            "--control-near-pass-score-floor",
            str(args.control_near_pass_score_floor),
            "--control-alt-profile",
            str(args.control_alt_profile),
            "--control-alt-arm-id",
            str(args.control_alt_arm_id),
            "--preflight-operational-gate",
            "--preflight-gate-arm",
            str(args.gate_arm),
            "--preflight-max-fallback-rate",
            str(args.max_fallback_rate),
            "--preflight-max-retry-rate",
            str(args.max_retry_rate),
            "--preflight-min-first-pass-accept-rate",
            str(args.min_first_pass_accept_rate),
            "--output-root",
            str(seed_out),
        ]
        if bool(args.enable_control_intent_risk_adaptation):
            command.append("--enable-control-intent-risk-adaptation")
        if bool(args.enable_control_latency_adaptation):
            command.append("--enable-control-latency-adaptation")
        if bool(args.enable_control_intent_focused_context):
            command.append("--enable-control-intent-focused-context")
        if bool(args.enable_control_near_pass):
            command.append("--enable-control-near-pass")
        if bool(args.disable_control_near_pass_block_high_risk):
            command.append("--disable-control-near-pass-block-high-risk")
        proc = subprocess.run(command, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"Seed run failed (seed={seed}, rc={proc.returncode})")

        run_dir = latest_subdir(seed_out)
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        paired = json.loads((run_dir / "paired_delta_significance.json").read_text(encoding="utf-8"))
        op = json.loads((run_dir / "operational_metrics.json").read_text(encoding="utf-8"))
        gate = json.loads((run_dir / "preflight_operational_gate.json").read_text(encoding="utf-8"))

        if gate_arm not in summary:
            raise KeyError(f"gate arm '{gate_arm}' missing in summary for run {run_dir}")
        if gate_arm not in op.get("arms", {}):
            raise KeyError(f"gate arm '{gate_arm}' missing in operational metrics for run {run_dir}")
        if paired_key not in paired:
            available = ", ".join(sorted(paired.keys()))
            raise KeyError(
                f"paired comparison '{paired_key}' missing for run {run_dir}; available: [{available}]"
            )

        controlled = summary[gate_arm]
        controlled_pair = paired[paired_key]
        op_arm = op["arms"][gate_arm]
        per_seed.append(
            {
                "seed": seed,
                "run_dir": str(run_dir),
                "gate_arm": gate_arm,
                "paired_key": paired_key,
                "gate_pass": bool(gate.get("pass")),
                "quality": {
                    "context_relevance": float(controlled["context_relevance"]["mean"]),
                    "persona_consistency": float(controlled["persona_consistency"]["mean"]),
                    "naturalness": float(controlled["naturalness"]["mean"]),
                    "overall_quality": float(controlled["overall_quality"]["mean"]),
                },
                "paired_delta_vs_raw": {
                    "context_relevance": float(controlled_pair["context_relevance"]["mean_delta"]),
                    "persona_consistency": float(controlled_pair["persona_consistency"]["mean_delta"]),
                    "naturalness": float(controlled_pair["naturalness"]["mean_delta"]),
                    "overall_quality": float(controlled_pair["overall_quality"]["mean_delta"]),
                },
                "operational": {
                    "fallback_rate": float(op_arm["fallback_rate"]),
                    "retry_rate": float(op_arm["retry_rate"]),
                    "first_pass_accept_rate": float(op_arm["first_pass_accept_rate"]),
                },
            }
        )

    quality_drifts = {
        "overall_quality": max_pairwise_abs([x["quality"]["overall_quality"] for x in per_seed]),
        "context_relevance": max_pairwise_abs([x["quality"]["context_relevance"] for x in per_seed]),
        "persona_consistency": max_pairwise_abs([x["quality"]["persona_consistency"] for x in per_seed]),
        "naturalness": max_pairwise_abs([x["quality"]["naturalness"] for x in per_seed]),
    }
    operational_drifts = {
        "fallback_rate": max_pairwise_abs([x["operational"]["fallback_rate"] for x in per_seed]),
        "retry_rate": max_pairwise_abs([x["operational"]["retry_rate"] for x in per_seed]),
    }

    checks = [
        {"name": "all_seed_gates_pass", "pass": all(x["gate_pass"] for x in per_seed)},
        {"name": "overall_drift", "pass": quality_drifts["overall_quality"] <= float(args.max_overall_drift)},
        {"name": "context_drift", "pass": quality_drifts["context_relevance"] <= float(args.max_context_drift)},
        {"name": "persona_drift", "pass": quality_drifts["persona_consistency"] <= float(args.max_persona_drift)},
        {"name": "naturalness_drift", "pass": quality_drifts["naturalness"] <= float(args.max_naturalness_drift)},
        {"name": "fallback_drift", "pass": operational_drifts["fallback_rate"] <= float(args.max_fallback_drift)},
        {"name": "retry_drift", "pass": operational_drifts["retry_rate"] <= float(args.max_retry_drift)},
    ]
    passed = all(bool(c["pass"]) for c in checks)

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": seeds,
        "gate_arm": gate_arm,
        "paired_key": paired_key,
        "thresholds": {
            "max_fallback_rate": float(args.max_fallback_rate),
            "max_retry_rate": float(args.max_retry_rate),
            "min_first_pass_accept_rate": float(args.min_first_pass_accept_rate),
            "max_overall_drift": float(args.max_overall_drift),
            "max_context_drift": float(args.max_context_drift),
            "max_persona_drift": float(args.max_persona_drift),
            "max_naturalness_drift": float(args.max_naturalness_drift),
            "max_fallback_drift": float(args.max_fallback_drift),
            "max_retry_drift": float(args.max_retry_drift),
        },
        "per_seed": per_seed,
        "drift": {
            "quality": quality_drifts,
            "operational": operational_drifts,
        },
        "checks": checks,
        "pass": passed,
    }
    write_json(preflight_dir / "preflight_report.json", report)

    md_lines: List[str] = []
    md_lines.append("# Control Preflight Report")
    md_lines.append("")
    md_lines.append(f"- Pass: `{passed}`")
    md_lines.append(f"- Seeds: `{', '.join(str(s) for s in seeds)}`")
    md_lines.append("")
    md_lines.append("## Drift Summary")
    md_lines.append("")
    md_lines.append(f"- Overall quality drift: `{quality_drifts['overall_quality']:.4f}`")
    md_lines.append(f"- Context relevance drift: `{quality_drifts['context_relevance']:.4f}`")
    md_lines.append(f"- Persona consistency drift: `{quality_drifts['persona_consistency']:.4f}`")
    md_lines.append(f"- Naturalness drift: `{quality_drifts['naturalness']:.4f}`")
    md_lines.append(f"- Fallback-rate drift: `{operational_drifts['fallback_rate']:.4f}`")
    md_lines.append(f"- Retry-rate drift: `{operational_drifts['retry_rate']:.4f}`")
    md_lines.append("")
    md_lines.append("## Check Results")
    md_lines.append("")
    for item in checks:
        md_lines.append(f"- `{item['name']}`: `{'PASS' if item['pass'] else 'FAIL'}`")
    (preflight_dir / "preflight_report.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Preflight report: {preflight_dir / 'preflight_report.json'}")
    print(f"Markdown report: {preflight_dir / 'preflight_report.md'}")
    if args.strict and not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

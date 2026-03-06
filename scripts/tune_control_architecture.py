#!/usr/bin/env python3
"""Constrained tuning for response-control architecture profiles.

Methodology choices:
- Random search over focused parameter subspace (effective baseline for HPO).
- Source-group-aware train/validation split to reduce template leakage.
- Multi-seed aggregation to reduce single-seed noise.
- Constrained objective: optimize quality deltas while penalizing operational regressions.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_SCRIPT = SCRIPT_DIR / "run_proposal_alignment_eval.py"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def latest_subdir(root: Path) -> Path:
    dirs = [p for p in root.iterdir() if p.is_dir()]
    if not dirs:
        raise FileNotFoundError(f"No run directory found under: {root}")
    return sorted(dirs, key=lambda p: p.name)[-1]


def parse_seeds(raw: str) -> List[int]:
    out: List[int] = []
    for chunk in str(raw).replace(";", ",").split(","):
        token = chunk.strip()
        if not token:
            continue
        out.append(int(token))
    dedup: List[int] = []
    seen = set()
    for s in out:
        if s in seen:
            continue
        seen.add(s)
        dedup.append(s)
    if not dedup:
        raise ValueError("No seeds provided.")
    return dedup


def scenario_source_id(row: Dict[str, Any]) -> str:
    sid = str(row.get("scenario_id", "")).strip()
    source_id = str(row.get("source_scenario_id", "")).strip()
    return source_id or sid


def split_by_source(
    rows: Sequence[Dict[str, Any]],
    train_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_source: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_source.setdefault(scenario_source_id(row), []).append(dict(row))
    source_ids = sorted(by_source.keys())
    rng = random.Random(seed)
    rng.shuffle(source_ids)
    cut = max(1, min(len(source_ids) - 1, int(round(len(source_ids) * train_ratio)))) if len(source_ids) > 1 else 1
    train_sources = set(source_ids[:cut])
    train_rows: List[Dict[str, Any]] = []
    valid_rows: List[Dict[str, Any]] = []
    for sid in source_ids:
        target = train_rows if sid in train_sources else valid_rows
        target.extend(by_source[sid])
    if not valid_rows:
        valid_rows = list(train_rows[-max(1, len(train_rows) // 3) :])
    return train_rows, valid_rows


def sample_candidate(rng: random.Random) -> Dict[str, Any]:
    return {
        "min_context_coverage": round(rng.uniform(0.28, 0.40), 3),
        "min_persona_coverage": round(rng.uniform(0.14, 0.24), 3),
        "relaxed_context_coverage": round(rng.uniform(0.14, 0.24), 3),
        "relaxed_persona_coverage": round(rng.uniform(0.07, 0.14), 3),
        "relaxed_candidate_score": round(rng.uniform(0.40, 0.52), 3),
        "adaptive_candidate_score": round(rng.uniform(0.30, 0.46), 3),
        "adaptive_context_coverage": round(rng.uniform(0.10, 0.20), 3),
        "adaptive_persona_coverage": round(rng.uniform(0.08, 0.15), 3),
        "adaptive_high_confidence_rewrites": 1,
        "adaptive_mid_confidence_rewrites": rng.randint(1, 3),
        "adaptive_low_confidence_rewrites": rng.randint(2, 4),
        "low_confidence_retry_requires_gain": True,
        "low_confidence_retry_min_score_gain": round(rng.uniform(0.008, 0.025), 3),
        "low_confidence_retry_min_coverage_gain": round(rng.uniform(0.015, 0.045), 3),
        "early_stop_score": round(rng.uniform(0.64, 0.76), 3),
        "intent_risk_adaptation_enabled": bool(rng.choice([True, False])),
        "latency_adaptation_enabled": bool(rng.choice([True, False])),
        "latency_relax_start_pressure": round(rng.uniform(0.45, 0.65), 3),
        "latency_relax_max_delta": round(rng.uniform(0.04, 0.12), 3),
        "low_risk_context_relax": round(rng.uniform(0.03, 0.08), 3),
        "low_risk_persona_relax": round(rng.uniform(0.02, 0.05), 3),
        "low_risk_candidate_score_relax": round(rng.uniform(0.02, 0.05), 3),
        "high_risk_context_tighten": round(rng.uniform(0.01, 0.05), 3),
        "high_risk_persona_tighten": round(rng.uniform(0.005, 0.03), 3),
        "high_risk_candidate_score_tighten": round(rng.uniform(0.005, 0.04), 3),
        "intent_focused_context_enabled": bool(rng.choice([True, False])),
        "intent_focus_min_keep": rng.randint(2, 4),
        "intent_focus_keep_ratio_low": round(rng.uniform(0.35, 0.55), 3),
        "intent_focus_keep_ratio_medium": round(rng.uniform(0.50, 0.75), 3),
        "intent_focus_keep_ratio_high": round(rng.uniform(0.90, 1.0), 3),
        "intent_focus_min_relevance": round(rng.uniform(0.12, 0.28), 3),
        "near_pass_enabled": bool(rng.choice([True, False])),
        "near_pass_max_context_gap": round(rng.uniform(0.025, 0.070), 3),
        "near_pass_max_persona_gap": round(rng.uniform(0.020, 0.060), 3),
        "near_pass_score_floor": round(rng.uniform(0.30, 0.42), 3),
        "near_pass_block_high_risk": bool(rng.choice([True, False])),
    }


def warm_start_candidates() -> List[Dict[str, Any]]:
    return [
        {
            "min_context_coverage": 0.31,
            "min_persona_coverage": 0.17,
            "relaxed_context_coverage": 0.16,
            "relaxed_persona_coverage": 0.08,
            "relaxed_candidate_score": 0.42,
            "adaptive_candidate_score": 0.35,
            "rewrite_candidates": 2,
            "adaptive_mid_confidence_rewrites": 1,
            "adaptive_low_confidence_rewrites": 2,
            "intent_risk_adaptation_enabled": True,
            "latency_adaptation_enabled": True,
            "latency_relax_start_pressure": 0.50,
            "latency_relax_max_delta": 0.08,
            "low_risk_context_relax": 0.06,
            "low_risk_persona_relax": 0.04,
            "low_risk_candidate_score_relax": 0.04,
            "high_risk_context_tighten": 0.02,
            "high_risk_persona_tighten": 0.01,
            "high_risk_candidate_score_tighten": 0.01,
            "intent_focused_context_enabled": True,
            "intent_focus_min_keep": 3,
            "intent_focus_keep_ratio_low": 0.42,
            "intent_focus_keep_ratio_medium": 0.62,
            "intent_focus_keep_ratio_high": 0.95,
            "intent_focus_min_relevance": 0.18,
            "near_pass_enabled": True,
            "near_pass_max_context_gap": 0.04,
            "near_pass_max_persona_gap": 0.03,
            "near_pass_score_floor": 0.35,
            "near_pass_block_high_risk": True,
        },
        {
            "min_context_coverage": 0.31,
            "min_persona_coverage": 0.16,
            "relaxed_context_coverage": 0.16,
            "relaxed_persona_coverage": 0.08,
            "relaxed_candidate_score": 0.42,
            "adaptive_candidate_score": 0.34,
            "rewrite_candidates": 2,
            "adaptive_high_confidence_rewrites": 1,
            "adaptive_mid_confidence_rewrites": 1,
            "adaptive_low_confidence_rewrites": 2,
            "low_confidence_retry_requires_gain": True,
            "low_confidence_retry_min_score_gain": 0.015,
            "low_confidence_retry_min_coverage_gain": 0.03,
            "intent_risk_adaptation_enabled": False,
            "latency_adaptation_enabled": True,
            "latency_relax_start_pressure": 0.55,
            "latency_relax_max_delta": 0.08,
            "intent_focused_context_enabled": False,
            "near_pass_enabled": False,
            "near_pass_block_high_risk": True,
        },
    ]


@dataclass
class EvalOutcome:
    run_dir: str
    overall_delta: float
    context_delta: float
    persona_delta: float
    naturalness_delta: float
    fallback_default: float
    retry_default: float
    first_pass_default: float
    fallback_alt: float
    retry_alt: float
    first_pass_alt: float


def mean_or_nan(vals: Sequence[float]) -> float:
    return statistics.fmean(vals) if vals else float("nan")


def run_eval_once(
    *,
    host: str,
    candidate_model: str,
    baseline_model: str,
    scenarios_path: Path,
    seed: int,
    max_scenarios: int,
    max_tokens: int,
    temperature: float,
    timeout_s: int,
    output_root: Path,
    overrides_path: Path,
    alt_arm_id: str,
) -> EvalOutcome:
    output_root.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--host",
        host,
        "--candidate-model",
        candidate_model,
        "--baseline-model",
        baseline_model,
        "--skip-external-baselines",
        "--disable-bertscore",
        "--scenarios",
        str(scenarios_path),
        "--max-scenarios",
        str(max_scenarios),
        "--repeats",
        "1",
        "--seed",
        str(seed),
        "--max-tokens",
        str(max_tokens),
        "--temperature",
        str(temperature),
        "--timeout-s",
        str(timeout_s),
        "--control-alt-profile",
        "custom",
        "--control-alt-arm-id",
        alt_arm_id,
        "--control-alt-overrides-file",
        str(overrides_path),
        "--target-arm",
        alt_arm_id,
        "--output-root",
        str(output_root),
    ]
    proc = subprocess.run(command, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Evaluation run failed (seed={seed}, rc={proc.returncode})")

    run_dir = latest_subdir(output_root)
    paired = json.loads((run_dir / "paired_delta_significance.json").read_text(encoding="utf-8"))
    op = json.loads((run_dir / "operational_metrics.json").read_text(encoding="utf-8"))
    comp = paired.get("controlled_alt_vs_controlled_default", {})
    if not comp:
        raise RuntimeError("Missing comparison 'controlled_alt_vs_controlled_default' in paired deltas.")

    arm_default = op.get("arms", {}).get("proposed_contextual_controlled", {})
    arm_alt = op.get("arms", {}).get(alt_arm_id, {})
    return EvalOutcome(
        run_dir=str(run_dir),
        overall_delta=float(comp.get("overall_quality", {}).get("mean_delta", float("nan"))),
        context_delta=float(comp.get("context_relevance", {}).get("mean_delta", float("nan"))),
        persona_delta=float(comp.get("persona_consistency", {}).get("mean_delta", float("nan"))),
        naturalness_delta=float(comp.get("naturalness", {}).get("mean_delta", float("nan"))),
        fallback_default=float(arm_default.get("fallback_rate", float("nan"))),
        retry_default=float(arm_default.get("retry_rate", float("nan"))),
        first_pass_default=float(arm_default.get("first_pass_accept_rate", float("nan"))),
        fallback_alt=float(arm_alt.get("fallback_rate", float("nan"))),
        retry_alt=float(arm_alt.get("retry_rate", float("nan"))),
        first_pass_alt=float(arm_alt.get("first_pass_accept_rate", float("nan"))),
    )


def summarize_outcomes(outcomes: Sequence[EvalOutcome]) -> Dict[str, float]:
    if not outcomes:
        return {}
    overall = mean_or_nan([o.overall_delta for o in outcomes])
    context = mean_or_nan([o.context_delta for o in outcomes])
    persona = mean_or_nan([o.persona_delta for o in outcomes])
    naturalness = mean_or_nan([o.naturalness_delta for o in outcomes])
    fallback_gap = mean_or_nan([o.fallback_alt - o.fallback_default for o in outcomes])
    retry_gap = mean_or_nan([o.retry_alt - o.retry_default for o in outcomes])
    first_pass_gap = mean_or_nan([o.first_pass_alt - o.first_pass_default for o in outcomes])
    return {
        "overall_delta": overall,
        "context_delta": context,
        "persona_delta": persona,
        "naturalness_delta": naturalness,
        "fallback_gap": fallback_gap,
        "retry_gap": retry_gap,
        "first_pass_gap": first_pass_gap,
    }


def score_summary(summary: Dict[str, float], args: argparse.Namespace) -> Dict[str, Any]:
    overall = float(summary.get("overall_delta", float("nan")))
    context = float(summary.get("context_delta", float("nan")))
    persona = float(summary.get("persona_delta", float("nan")))
    naturalness = float(summary.get("naturalness_delta", float("nan")))
    fallback_gap = float(summary.get("fallback_gap", float("nan")))
    retry_gap = float(summary.get("retry_gap", float("nan")))
    first_pass_gap = float(summary.get("first_pass_gap", float("nan")))

    feasible = (
        overall >= float(args.min_overall_delta)
        and context >= float(args.min_context_delta)
        and persona >= float(args.min_persona_delta)
        and fallback_gap <= float(args.max_fallback_increase)
        and retry_gap <= float(args.max_retry_increase)
        and first_pass_gap >= -float(args.max_first_pass_drop)
    )

    penalty = 0.0
    penalty += max(0.0, fallback_gap - float(args.max_fallback_increase)) * 2.0
    penalty += max(0.0, retry_gap - float(args.max_retry_increase)) * 1.25
    penalty += max(0.0, (-first_pass_gap) - float(args.max_first_pass_drop)) * 1.5
    penalty += max(0.0, float(args.min_context_delta) - context) * 1.25
    penalty += max(0.0, float(args.min_persona_delta) - persona) * 1.0
    penalty += max(0.0, float(args.min_overall_delta) - overall) * 2.0

    objective = (
        overall
        + 0.20 * context
        + 0.20 * persona
        + 0.08 * naturalness
        - 0.60 * max(0.0, fallback_gap)
        - 0.35 * max(0.0, retry_gap)
        - 0.45 * max(0.0, -first_pass_gap)
        - penalty
    )
    return {
        "feasible": bool(feasible),
        "objective": float(objective),
        "penalty": float(penalty),
    }


def candidate_sort_key(item: Dict[str, Any]) -> Tuple[int, float]:
    score = item.get("validation_score") or item.get("train_score") or {}
    return (1 if bool(score.get("feasible")) else 0, float(score.get("objective", -1e9)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune alternate response-control profile with constraints.")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--candidate-model", default="elara-npc:latest")
    parser.add_argument("--baseline-model", default="phi3:mini")
    parser.add_argument("--scenarios", default="data/proposal_eval_scenarios_112_diverse.jsonl")
    parser.add_argument("--output-root", default="artifacts/proposal_control_tuning/auto_tune")
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--train-ratio", type=float, default=0.67)
    parser.add_argument("--train-seeds", default="19")
    parser.add_argument("--valid-seeds", default="29")
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--topk-validate", type=int, default=2)
    parser.add_argument("--train-max-scenarios", type=int, default=24)
    parser.add_argument("--valid-max-scenarios", type=int, default=24)
    parser.add_argument("--max-tokens", type=int, default=72)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout-s", type=int, default=120)
    parser.add_argument("--alt-arm-id", default="proposed_contextual_controlled_tuned")

    parser.add_argument("--min-overall-delta", type=float, default=0.0)
    parser.add_argument("--min-context-delta", type=float, default=-0.005)
    parser.add_argument("--min-persona-delta", type=float, default=-0.005)
    parser.add_argument("--max-fallback-increase", type=float, default=0.03)
    parser.add_argument("--max-retry-increase", type=float, default=0.04)
    parser.add_argument("--max-first-pass-drop", type=float, default=0.04)

    args = parser.parse_args()

    scenarios_path = Path(args.scenarios)
    if not scenarios_path.exists():
        raise FileNotFoundError(f"Scenario file not found: {scenarios_path}")

    run_root = Path(args.output_root) / utc_stamp()
    run_root.mkdir(parents=True, exist_ok=True)
    data_dir = run_root / "data"
    train_runs_root = run_root / "train_runs"
    valid_runs_root = run_root / "valid_runs"
    candidate_dir = run_root / "candidates"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_runs_root.mkdir(parents=True, exist_ok=True)
    valid_runs_root.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)

    all_rows = read_jsonl(scenarios_path)
    if len(all_rows) < 8:
        raise ValueError("Need at least 8 scenarios for split tuning.")

    train_rows, valid_rows = split_by_source(all_rows, train_ratio=float(args.train_ratio), seed=int(args.seed))
    train_path = data_dir / "train.jsonl"
    valid_path = data_dir / "valid.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(valid_path, valid_rows)

    train_seeds = parse_seeds(str(args.train_seeds))
    valid_seeds = parse_seeds(str(args.valid_seeds))
    rng = random.Random(int(args.seed))

    search_records: List[Dict[str, Any]] = []
    print(f"[tune] run_root={run_root}")
    print(f"[tune] train_scenarios={len(train_rows)} valid_scenarios={len(valid_rows)}")

    candidate_specs: List[Tuple[int, Dict[str, Any]]] = [(0, {})]
    trial_budget = max(1, int(args.trials))
    next_trial_idx = 1
    for warm in warm_start_candidates():
        if next_trial_idx > trial_budget:
            break
        candidate_specs.append((next_trial_idx, warm))
        next_trial_idx += 1
    while next_trial_idx <= trial_budget:
        candidate_specs.append((next_trial_idx, sample_candidate(rng)))
        next_trial_idx += 1

    for trial_idx, overrides in candidate_specs:
        overrides_path = candidate_dir / f"trial_{trial_idx:03d}.json"
        write_json(overrides_path, overrides)
        print(f"[tune][train] trial={trial_idx}/{len(candidate_specs) - 1} (0=baseline)")

        outcomes: List[EvalOutcome] = []
        for seed in train_seeds:
            trial_out_root = train_runs_root / f"trial_{trial_idx:03d}" / f"seed_{seed}"
            outcome = run_eval_once(
                host=str(args.host),
                candidate_model=str(args.candidate_model),
                baseline_model=str(args.baseline_model),
                scenarios_path=train_path,
                seed=int(seed),
                max_scenarios=int(args.train_max_scenarios),
                max_tokens=int(args.max_tokens),
                temperature=float(args.temperature),
                timeout_s=int(args.timeout_s),
                output_root=trial_out_root,
                overrides_path=overrides_path,
                alt_arm_id=str(args.alt_arm_id),
            )
            outcomes.append(outcome)

        summary = summarize_outcomes(outcomes)
        train_score = score_summary(summary, args)
        record = {
            "trial_index": trial_idx,
            "overrides": overrides,
            "overrides_path": str(overrides_path),
            "train_outcomes": [o.__dict__ for o in outcomes],
            "train_summary": summary,
            "train_score": train_score,
        }
        search_records.append(record)
        print(
            f"[tune][train] trial={trial_idx} objective={train_score['objective']:.4f} "
            f"feasible={train_score['feasible']} overall_delta={summary.get('overall_delta', float('nan')):.4f}"
        )

    ranked = sorted(search_records, key=candidate_sort_key, reverse=True)
    topk = max(1, min(int(args.topk_validate), len(ranked)))
    validate_pool = ranked[:topk]

    for item in validate_pool:
        trial_idx = int(item["trial_index"])
        overrides_path = Path(str(item["overrides_path"]))
        outcomes: List[EvalOutcome] = []
        print(f"[tune][valid] trial={trial_idx}")
        for seed in valid_seeds:
            trial_out_root = valid_runs_root / f"trial_{trial_idx:03d}" / f"seed_{seed}"
            outcome = run_eval_once(
                host=str(args.host),
                candidate_model=str(args.candidate_model),
                baseline_model=str(args.baseline_model),
                scenarios_path=valid_path,
                seed=int(seed),
                max_scenarios=int(args.valid_max_scenarios),
                max_tokens=int(args.max_tokens),
                temperature=float(args.temperature),
                timeout_s=int(args.timeout_s),
                output_root=trial_out_root,
                overrides_path=overrides_path,
                alt_arm_id=str(args.alt_arm_id),
            )
            outcomes.append(outcome)
        valid_summary = summarize_outcomes(outcomes)
        valid_score = score_summary(valid_summary, args)
        item["validation_outcomes"] = [o.__dict__ for o in outcomes]
        item["validation_summary"] = valid_summary
        item["validation_score"] = valid_score
        print(
            f"[tune][valid] trial={trial_idx} objective={valid_score['objective']:.4f} "
            f"feasible={valid_score['feasible']} overall_delta={valid_summary.get('overall_delta', float('nan')):.4f}"
        )

    final_ranked = sorted(search_records, key=candidate_sort_key, reverse=True)
    recommended = final_ranked[0] if final_ranked else None
    if recommended is None:
        raise RuntimeError("No candidate produced.")

    recommended_path = run_root / "recommended_overrides.json"
    write_json(recommended_path, recommended.get("overrides", {}))

    report = {
        "generated_utc": utc_iso(),
        "run_root": str(run_root),
        "config": {
            "host": str(args.host),
            "candidate_model": str(args.candidate_model),
            "baseline_model": str(args.baseline_model),
            "scenarios": str(scenarios_path),
            "train_ratio": float(args.train_ratio),
            "train_seeds": train_seeds,
            "valid_seeds": valid_seeds,
            "trials": int(args.trials),
            "topk_validate": int(args.topk_validate),
            "train_max_scenarios": int(args.train_max_scenarios),
            "valid_max_scenarios": int(args.valid_max_scenarios),
        },
        "split": {
            "all_count": len(all_rows),
            "train_count": len(train_rows),
            "valid_count": len(valid_rows),
            "train_path": str(train_path),
            "valid_path": str(valid_path),
        },
        "constraints": {
            "min_overall_delta": float(args.min_overall_delta),
            "min_context_delta": float(args.min_context_delta),
            "min_persona_delta": float(args.min_persona_delta),
            "max_fallback_increase": float(args.max_fallback_increase),
            "max_retry_increase": float(args.max_retry_increase),
            "max_first_pass_drop": float(args.max_first_pass_drop),
        },
        "candidates": final_ranked,
        "recommended": {
            "trial_index": int(recommended["trial_index"]),
            "overrides": recommended["overrides"],
            "overrides_path": str(recommended_path),
            "train_score": recommended.get("train_score", {}),
            "train_summary": recommended.get("train_summary", {}),
            "validation_score": recommended.get("validation_score", {}),
            "validation_summary": recommended.get("validation_summary", {}),
        },
        "full112_command": [
            sys.executable,
            str(SCRIPT_DIR / "run_proposal_alignment_eval_batched.py"),
            "--host",
            str(args.host),
            "--candidate-model",
            str(args.candidate_model),
            "--baseline-model",
            str(args.baseline_model),
            "--scenarios",
            str(scenarios_path),
            "--batch-size",
            "28",
            "--repeats",
            "1",
            "--max-tokens",
            str(args.max_tokens),
            "--temperature",
            str(args.temperature),
            "--control-alt-profile",
            "custom",
            "--control-alt-arm-id",
            str(args.alt_arm_id),
            "--control-alt-overrides-file",
            str(recommended_path),
            "--output-root",
            str(run_root / "full112_recommended"),
        ],
    }
    write_json(run_root / "tuning_report.json", report)

    lines: List[str] = []
    lines.append("# Control Architecture Tuning Report")
    lines.append("")
    lines.append(f"- Run root: `{run_root}`")
    lines.append(f"- Train scenarios: `{len(train_rows)}`")
    lines.append(f"- Validation scenarios: `{len(valid_rows)}`")
    lines.append(f"- Trials: `{int(args.trials)}`")
    lines.append("")
    rec = report["recommended"]
    lines.append("## Recommended Candidate")
    lines.append(f"- Trial index: `{rec['trial_index']}`")
    lines.append(f"- Overrides file: `{recommended_path}`")
    lines.append(f"- Train objective: `{rec.get('train_score', {}).get('objective', float('nan')):.4f}`")
    lines.append(f"- Validation objective: `{rec.get('validation_score', {}).get('objective', float('nan')):.4f}`")
    lines.append("")
    lines.append("## Overrides")
    lines.append("```json")
    lines.append(json.dumps(rec["overrides"], indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")
    lines.append("## Full 112 Command")
    lines.append("```bash")
    lines.append(" ".join(report["full112_command"]))
    lines.append("```")
    (run_root / "tuning_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Tuning report: {run_root / 'tuning_report.json'}")
    print(f"Markdown report: {run_root / 'tuning_report.md'}")
    print(f"Recommended overrides: {recommended_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare response-control architecture profiles against default control."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_SCRIPT = SCRIPT_DIR / "run_proposal_alignment_eval.py"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def parse_csv(raw: str) -> List[str]:
    items = [x.strip() for x in str(raw).replace(";", ",").split(",") if x.strip()]
    dedup: List[str] = []
    seen = set()
    for item in items:
        low = item.lower()
        if low in seen:
            continue
        seen.add(low)
        dedup.append(item)
    return dedup


def parse_seeds(raw: str) -> List[int]:
    return [int(x) for x in parse_csv(raw)]


def latest_subdir(root: Path) -> Path:
    dirs = [p for p in root.iterdir() if p.is_dir()]
    if not dirs:
        raise FileNotFoundError(f"No run directory under: {root}")
    return sorted(dirs, key=lambda p: p.name)[-1]


def mean(values: List[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


@dataclass
class ProfileSeedOutcome:
    seed: int
    run_dir: str
    delta_context: float
    delta_persona: float
    delta_naturalness: float
    delta_overall: float
    default_fallback: float
    default_retry: float
    default_first_pass: float
    alt_fallback: float
    alt_retry: float
    alt_first_pass: float


def run_one(
    *,
    profile: str,
    seed: int,
    host: str,
    candidate_model: str,
    baseline_model: str,
    baseline_models: str,
    scenarios: str,
    max_scenarios: int,
    max_tokens: int,
    temperature: float,
    timeout_s: int,
    min_arm_success_rate: float,
    output_root: Path,
    alt_arm_id: str,
) -> ProfileSeedOutcome:
    output_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--host",
        host,
        "--candidate-model",
        candidate_model,
        "--baseline-model",
        baseline_model,
        "--baseline-models",
        baseline_models,
        "--scenarios",
        scenarios,
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
        "--min-arm-success-rate",
        str(min_arm_success_rate),
        "--control-alt-profile",
        profile,
        "--control-alt-arm-id",
        alt_arm_id,
        "--skip-external-baselines",
        "--disable-bertscore",
        "--output-root",
        str(output_root),
    ]
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"profile={profile} seed={seed} failed with rc={proc.returncode}")

    run_dir = latest_subdir(output_root)
    paired = json.loads((run_dir / "paired_delta_significance.json").read_text(encoding="utf-8"))
    op = json.loads((run_dir / "operational_metrics.json").read_text(encoding="utf-8"))

    comp = paired.get("controlled_alt_vs_controlled_default", {})
    if not comp:
        raise RuntimeError(f"Missing controlled_alt_vs_controlled_default in {run_dir}")
    arm_default = op.get("arms", {}).get("proposed_contextual_controlled", {})
    arm_alt = op.get("arms", {}).get(alt_arm_id, {})
    return ProfileSeedOutcome(
        seed=seed,
        run_dir=str(run_dir),
        delta_context=float(comp.get("context_relevance", {}).get("mean_delta", float("nan"))),
        delta_persona=float(comp.get("persona_consistency", {}).get("mean_delta", float("nan"))),
        delta_naturalness=float(comp.get("naturalness", {}).get("mean_delta", float("nan"))),
        delta_overall=float(comp.get("overall_quality", {}).get("mean_delta", float("nan"))),
        default_fallback=float(arm_default.get("fallback_rate", float("nan"))),
        default_retry=float(arm_default.get("retry_rate", float("nan"))),
        default_first_pass=float(arm_default.get("first_pass_accept_rate", float("nan"))),
        alt_fallback=float(arm_alt.get("fallback_rate", float("nan"))),
        alt_retry=float(arm_alt.get("retry_rate", float("nan"))),
        alt_first_pass=float(arm_alt.get("first_pass_accept_rate", float("nan"))),
    )


def run_one_with_retries(
    *,
    profile: str,
    seed: int,
    host: str,
    candidate_model: str,
    baseline_model: str,
    baseline_models: str,
    scenarios: str,
    max_scenarios: int,
    max_tokens: int,
    temperature: float,
    timeout_s: int,
    min_arm_success_rate: float,
    output_root: Path,
    alt_arm_id: str,
    retries: int,
) -> ProfileSeedOutcome:
    last_error: Optional[Exception] = None
    attempts = max(1, int(retries))
    for attempt in range(1, attempts + 1):
        try:
            return run_one(
                profile=profile,
                seed=seed,
                host=host,
                candidate_model=candidate_model,
                baseline_model=baseline_model,
                baseline_models=baseline_models,
                scenarios=scenarios,
                max_scenarios=max_scenarios,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout_s=timeout_s,
                min_arm_success_rate=min_arm_success_rate,
                output_root=output_root / f"attempt_{attempt}",
                alt_arm_id=alt_arm_id,
            )
        except Exception as exc:  # pragma: no cover - defensive retry wrapper
            last_error = exc
            if attempt < attempts:
                print(
                    f"  retrying profile={profile} seed={seed} "
                    f"attempt={attempt}/{attempts} after error: {exc}"
                )
    if last_error is None:
        raise RuntimeError(f"profile={profile} seed={seed} failed with unknown error")
    raise RuntimeError(str(last_error))


def summarize_profile(outcomes: List[ProfileSeedOutcome]) -> Dict[str, Any]:
    d_context = [o.delta_context for o in outcomes]
    d_persona = [o.delta_persona for o in outcomes]
    d_naturalness = [o.delta_naturalness for o in outcomes]
    d_overall = [o.delta_overall for o in outcomes]
    gap_fallback = [o.alt_fallback - o.default_fallback for o in outcomes]
    gap_retry = [o.alt_retry - o.default_retry for o in outcomes]
    gap_first = [o.alt_first_pass - o.default_first_pass for o in outcomes]
    return {
        "delta_context_mean": mean(d_context),
        "delta_persona_mean": mean(d_persona),
        "delta_naturalness_mean": mean(d_naturalness),
        "delta_overall_mean": mean(d_overall),
        "fallback_gap_mean": mean(gap_fallback),
        "retry_gap_mean": mean(gap_retry),
        "first_pass_gap_mean": mean(gap_first),
        "quality_score": (
            mean(d_overall) + 0.20 * mean(d_context) + 0.20 * mean(d_persona) + 0.10 * mean(d_naturalness)
        ),
        "operational_score": (
            (-1.0 * mean(gap_fallback)) + (-0.6 * mean(gap_retry)) + (0.8 * mean(gap_first))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare multiple control architecture profiles.")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--candidate-model", default="elara-npc:latest")
    parser.add_argument("--baseline-model", default="phi3:mini")
    parser.add_argument("--baseline-models", default="phi3:mini")
    parser.add_argument("--scenarios", default="data/proposal_eval_scenarios_112_diverse.jsonl")
    parser.add_argument("--profiles", default="runtime_optimized,hybrid_balanced,intent_focus_adaptive,blend_balanced")
    parser.add_argument("--seeds", default="29,31")
    parser.add_argument("--max-scenarios", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=72)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout-s", type=int, default=120)
    parser.add_argument("--min-arm-success-rate", type=float, default=0.0)
    parser.add_argument("--retries-per-seed", type=int, default=2)
    parser.add_argument("--min-successful-seeds", type=int, default=1)
    parser.add_argument("--control-alt-arm-id", default="proposed_contextual_controlled_alt")
    parser.add_argument("--output-root", default="artifacts/proposal_control_tuning/architecture_compare")
    args = parser.parse_args()

    profiles = [p.lower() for p in parse_csv(args.profiles) if p.lower() != "none"]
    seeds = parse_seeds(args.seeds)
    if not profiles:
        raise ValueError("No profiles provided.")
    if not seeds:
        raise ValueError("No seeds provided.")

    run_root = Path(args.output_root) / utc_stamp()
    run_root.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "host": str(args.host),
            "candidate_model": str(args.candidate_model),
            "baseline_model": str(args.baseline_model),
            "baseline_models": str(args.baseline_models),
            "scenarios": str(args.scenarios),
            "profiles": profiles,
            "seeds": seeds,
            "max_scenarios": int(args.max_scenarios),
            "max_tokens": int(args.max_tokens),
            "temperature": float(args.temperature),
            "timeout_s": int(args.timeout_s),
            "min_arm_success_rate": float(args.min_arm_success_rate),
        },
        "profiles": {},
    }

    for profile in profiles:
        profile_root = run_root / profile
        outcomes: List[ProfileSeedOutcome] = []
        print(f"[compare] profile={profile}")
        failed_seeds: List[Dict[str, Any]] = []
        for seed in seeds:
            seed_root = profile_root / f"seed_{seed}"
            try:
                outcome = run_one_with_retries(
                    profile=profile,
                    seed=seed,
                    host=str(args.host),
                    candidate_model=str(args.candidate_model),
                    baseline_model=str(args.baseline_model),
                    baseline_models=str(args.baseline_models),
                    scenarios=str(args.scenarios),
                    max_scenarios=int(args.max_scenarios),
                    max_tokens=int(args.max_tokens),
                    temperature=float(args.temperature),
                    timeout_s=int(args.timeout_s),
                    min_arm_success_rate=float(args.min_arm_success_rate),
                    output_root=seed_root,
                    alt_arm_id=str(args.control_alt_arm_id),
                    retries=int(args.retries_per_seed),
                )
                outcomes.append(outcome)
            except Exception as exc:
                failed_seeds.append({"seed": seed, "error": str(exc)})
                print(f"  failed profile={profile} seed={seed}: {exc}")
        if len(outcomes) < max(1, int(args.min_successful_seeds)):
            raise RuntimeError(
                f"profile={profile} has only {len(outcomes)} successful seeds "
                f"(required {max(1, int(args.min_successful_seeds))})"
            )
        summary = summarize_profile(outcomes)
        results["profiles"][profile] = {
            "summary": summary,
            "per_seed": [o.__dict__ for o in outcomes],
            "failed_seeds": failed_seeds,
        }
        print(
            f"  quality={summary['quality_score']:.4f} operational={summary['operational_score']:.4f} "
            f"overall_delta={summary['delta_overall_mean']:.4f} fallback_gap={summary['fallback_gap_mean']:.4f}"
        )

    ranked = sorted(
        [
            {
                "profile": profile,
                "quality_score": float(data["summary"]["quality_score"]),
                "operational_score": float(data["summary"]["operational_score"]),
                "delta_overall_mean": float(data["summary"]["delta_overall_mean"]),
                "fallback_gap_mean": float(data["summary"]["fallback_gap_mean"]),
                "retry_gap_mean": float(data["summary"]["retry_gap_mean"]),
                "first_pass_gap_mean": float(data["summary"]["first_pass_gap_mean"]),
            }
            for profile, data in results["profiles"].items()
        ],
        key=lambda x: (x["quality_score"], x["operational_score"]),
        reverse=True,
    )
    results["ranking"] = ranked
    write_json(run_root / "architecture_compare.json", results)

    lines: List[str] = []
    lines.append("# Control Architecture Comparison")
    lines.append("")
    lines.append(f"- Run root: `{run_root}`")
    lines.append(f"- Profiles: `{', '.join(profiles)}`")
    lines.append(f"- Seeds: `{', '.join(str(s) for s in seeds)}`")
    lines.append("")
    lines.append("| Profile | Quality Score | Operational Score | Delta OQ | Fallback Gap | Retry Gap | First-pass Gap |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in ranked:
        lines.append(
            "| "
            + f"{row['profile']} | "
            + f"{row['quality_score']:.4f} | "
            + f"{row['operational_score']:.4f} | "
            + f"{row['delta_overall_mean']:.4f} | "
            + f"{row['fallback_gap_mean']:.4f} | "
            + f"{row['retry_gap_mean']:.4f} | "
            + f"{row['first_pass_gap_mean']:.4f} |"
        )
    (run_root / "architecture_compare.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {run_root / 'architecture_compare.json'}")
    print(f"Saved: {run_root / 'architecture_compare.md'}")


if __name__ == "__main__":
    main()

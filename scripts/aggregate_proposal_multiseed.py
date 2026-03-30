#!/usr/bin/env python3
"""Aggregate proposal metrics across multiple run IDs (multi-seed evidence)."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def parse_run_ids(raw: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for token in (x.strip() for x in raw.split(",")):
        if not token:
            continue
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def maybe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def summarize(values: List[float]) -> Dict[str, Any]:
    vals = [v for v in values if not math.isnan(v)]
    n = len(vals)
    if n == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
        }
    mean = statistics.fmean(vals)
    std = statistics.stdev(vals) if n > 1 else 0.0
    half = 1.96 * (std / math.sqrt(n)) if n > 1 else float("nan")
    return {
        "count": n,
        "mean": mean,
        "std": std,
        "min": min(vals),
        "max": max(vals),
        "ci95_low": (mean - half) if not math.isnan(half) else float("nan"),
        "ci95_high": (mean + half) if not math.isnan(half) else float("nan"),
    }


def build_markdown(payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Proposal Multi-Seed Aggregate")
    lines.append("")
    lines.append(f"- Generated: `{payload.get('generated_utc', '')}`")
    lines.append(f"- Proposal root: `{payload.get('proposal_root', '')}`")
    lines.append(f"- Target arm: `{payload.get('target_arm', '')}`")
    lines.append(f"- Runs included: `{len(payload.get('runs', []))}`")
    lines.append("")

    lines.append("## Included Runs")
    for row in payload.get("runs", []):
        lines.append(f"- `{row.get('run_id', '')}` -> `{row.get('run_dir', '')}`")
    lines.append("")

    lines.append("## Arm Metric Aggregates")
    lines.append("| Metric | Mean | Std | 95% CI | Min | Max | N |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for metric, stats in payload.get("metrics", {}).items():
        lines.append(
            "| {} | {:.6f} | {:.6f} | ({:.6f}, {:.6f}) | {:.6f} | {:.6f} | {} |".format(
                metric,
                maybe_float(stats.get("mean")),
                maybe_float(stats.get("std")),
                maybe_float(stats.get("ci95_low")),
                maybe_float(stats.get("ci95_high")),
                maybe_float(stats.get("min")),
                maybe_float(stats.get("max")),
                int(stats.get("count", 0) or 0),
            )
        )
    lines.append("")

    lines.append("## Operational Aggregates")
    lines.append("| Metric | Mean | Std | 95% CI | Min | Max | N |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for metric, stats in payload.get("operational", {}).items():
        lines.append(
            "| {} | {:.6f} | {:.6f} | ({:.6f}, {:.6f}) | {:.6f} | {:.6f} | {} |".format(
                metric,
                maybe_float(stats.get("mean")),
                maybe_float(stats.get("std")),
                maybe_float(stats.get("ci95_low")),
                maybe_float(stats.get("ci95_high")),
                maybe_float(stats.get("min")),
                maybe_float(stats.get("max")),
                int(stats.get("count", 0) or 0),
            )
        )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate proposal run metrics across multiple seeds.")
    parser.add_argument("--proposal-root", default="storage/artifacts/proposal")
    parser.add_argument("--run-ids", required=True, help="Comma-separated proposal run IDs.")
    parser.add_argument("--target-arm", default="proposed_contextual_controlled")
    parser.add_argument(
        "--metrics",
        default=(
            "overall_quality,context_relevance,persona_consistency,naturalness,"
            "quest_state_correctness,lore_consistency,multi_turn_contradiction_safety,"
            "objective_completion_support,gameplay_usefulness,time_pressure_acceptability"
        ),
    )
    parser.add_argument("--output-json", default="storage/artifacts/proposal_multiseed/multiseed_aggregate.json")
    parser.add_argument("--output-md", default="storage/artifacts/proposal_multiseed/multiseed_aggregate.md")
    args = parser.parse_args()

    proposal_root = Path(args.proposal_root)
    run_ids = parse_run_ids(str(args.run_ids))
    if not run_ids:
        raise ValueError("No run IDs provided.")

    metric_names = parse_run_ids(str(args.metrics))
    op_metric_names = ["fallback_rate", "retry_rate", "first_pass_accept_rate", "timeout_rate", "error_rate"]

    run_rows: List[Dict[str, Any]] = []
    by_metric: Dict[str, List[float]] = {m: [] for m in metric_names}
    by_op_metric: Dict[str, List[float]] = {m: [] for m in op_metric_names}

    for run_id in run_ids:
        run_dir = proposal_root / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Proposal run not found: {run_dir}")

        summary = read_json(run_dir / "summary.json")
        op = read_json(run_dir / "operational_metrics.json")

        arm_payload = summary.get(str(args.target_arm), {})
        if not isinstance(arm_payload, dict):
            raise ValueError(f"Target arm missing in summary for run {run_id}: {args.target_arm}")
        op_arm = op.get("arms", {}).get(str(args.target_arm), {})

        run_row: Dict[str, Any] = {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "metrics": {},
            "operational": {},
        }

        for metric in metric_names:
            metric_mean = maybe_float(arm_payload.get(metric, {}).get("mean"))
            run_row["metrics"][metric] = metric_mean
            by_metric.setdefault(metric, []).append(metric_mean)

        for metric in op_metric_names:
            metric_val = maybe_float(op_arm.get(metric))
            run_row["operational"][metric] = metric_val
            by_op_metric.setdefault(metric, []).append(metric_val)

        run_rows.append(run_row)

    payload = {
        "generated_utc": utc_iso(),
        "proposal_root": str(proposal_root),
        "target_arm": str(args.target_arm),
        "runs": run_rows,
        "metrics": {k: summarize(v) for k, v in by_metric.items()},
        "operational": {k: summarize(v) for k, v in by_op_metric.items()},
    }

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    write_json(out_json, payload)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(build_markdown(payload), encoding="utf-8")

    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Attach human-eval summary/report to an existing proposal run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import run_proposal_alignment_eval as evalmod  # type: ignore


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Attach human evaluation outputs to an existing proposal run.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--human-eval-file", required=True)
    parser.add_argument(
        "--human-eval-metrics",
        default="context_relevance,persona_consistency,naturalness,overall_quality",
    )
    parser.add_argument("--human-eval-scale-max", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=14001)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    human_file = Path(args.human_eval_file)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")
    if not human_file.exists():
        raise FileNotFoundError(f"Human eval file not found: {human_file}")

    comparison_plan = read_json(run_dir / "comparison_plan.json")
    target_arm = str(comparison_plan.get("target_arm_for_external", "proposed_contextual_controlled"))
    baseline_arms = [str(x) for x in comparison_plan.get("baseline_arm_ids", [])]

    rows = evalmod.read_human_eval_rows(human_file)
    metrics = evalmod.parse_list_arg(str(args.human_eval_metrics))
    summary = evalmod.analyze_human_eval(
        rows=rows,
        metrics=metrics,
        scale_max=float(args.human_eval_scale_max),
        target_arm=target_arm,
        baseline_arms=baseline_arms,
        seed=int(args.seed),
    )

    write_json(run_dir / "human_eval_summary.json", summary)
    evalmod.render_human_eval_report(run_dir / "human_eval_report.md", summary)

    manifest = {
        "human_eval_file": str(human_file),
        "metrics": metrics,
        "scale_max": float(args.human_eval_scale_max),
        "row_count": int(summary.get("row_count", 0)),
        "target_arm": target_arm,
        "baseline_arms": baseline_arms,
    }
    write_json(run_dir / "human_eval_attachment_manifest.json", manifest)

    print(f"Attached human eval to run: {run_dir}")
    print(f"  - {run_dir / 'human_eval_summary.json'}")
    print(f"  - {run_dir / 'human_eval_report.md'}")


if __name__ == "__main__":
    main()


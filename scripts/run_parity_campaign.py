#!/usr/bin/env python3
"""Run parity verification campaign across multiple harness/UE5 trace pairs."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def discover_pairs(harness_root: Path, ue5_root: Path, file_name: str) -> List[Dict[str, str]]:
    pairs: List[Dict[str, str]] = []
    if not harness_root.exists() or not ue5_root.exists():
        return pairs
    harness_runs = [p for p in harness_root.iterdir() if p.is_dir()]
    for run in sorted(harness_runs, key=lambda p: p.name):
        harness_file = run / "responses" / file_name
        ue5_file = ue5_root / run.name / "responses" / file_name
        if harness_file.exists() and ue5_file.exists():
            pairs.append(
                {
                    "run_id": run.name,
                    "harness_jsonl": str(harness_file),
                    "ue5_jsonl": str(ue5_file),
                }
            )
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping-json", default="", help="Optional explicit mapping list [{run_id,harness_jsonl,ue5_jsonl}]")
    parser.add_argument("--harness-root", default="storage/artifacts/proposal")
    parser.add_argument("--ue5-root", default="storage/artifacts/ue5_proposal")
    parser.add_argument("--response-file", default="proposed_contextual_controlled.jsonl")
    parser.add_argument("--min-scenarios", type=int, default=20)
    parser.add_argument("--min-equivalence-rate", type=float, default=0.98)
    parser.add_argument("--near-match-jaccard", type=float, default=0.92)
    parser.add_argument("--require-min-pass-rate", type=float, default=0.0)
    parser.add_argument("--previous-summary-json", default="")
    parser.add_argument("--fail-on-regression", action="store_true")
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument("--output-root", default="storage/artifacts/publication_profiles/parity_campaign")
    args = parser.parse_args()

    if str(args.mapping_json).strip():
        mapping = json.loads(Path(args.mapping_json).read_text(encoding="utf-8-sig"))
        pairs = [dict(x) for x in mapping if isinstance(x, dict)]
    else:
        pairs = discover_pairs(
            Path(args.harness_root),
            Path(args.ue5_root),
            str(args.response_file),
        )

    if (not pairs) and (not bool(args.allow_empty)):
        raise RuntimeError("No parity pairs found. Provide --mapping-json or matching harness/ue5 roots.")

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for pair in pairs:
        run_id = str(pair.get("run_id", "run"))
        h = str(pair.get("harness_jsonl", ""))
        u = str(pair.get("ue5_jsonl", ""))
        out_json = out_root / f"{run_id}.json"
        out_md = out_root / f"{run_id}.md"
        cmd = [
            sys.executable,
            "scripts/run_parity_verification_protocol.py",
            "--harness-jsonl",
            h,
            "--ue5-jsonl",
            u,
            "--min-scenarios",
            str(int(args.min_scenarios)),
            "--min-equivalence-rate",
            str(float(args.min_equivalence_rate)),
            "--near-match-jaccard",
            str(float(args.near_match_jaccard)),
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
        ]
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            rows.append(
                {
                    "run_id": run_id,
                    "pass": False,
                    "error": (proc.stderr or proc.stdout or "").strip()[-2000:],
                    "output_json": str(out_json),
                }
            )
            continue
        payload = json.loads(out_json.read_text(encoding="utf-8"))
        rows.append(
            {
                "run_id": run_id,
                "pass": bool(payload.get("pass", False)),
                "equivalence_rate": float(payload.get("metrics", {}).get("equivalence_rate", float("nan"))),
                "compared_scenarios": int(payload.get("counts", {}).get("compared_scenarios", 0)),
                "compared_turns": int(payload.get("counts", {}).get("compared_turns", 0)),
                "output_json": str(out_json),
            }
        )

    pass_count = sum(1 for r in rows if bool(r.get("pass")))
    summary = {
        "pair_count": len(rows),
        "pass_count": pass_count,
        "pass_rate": (pass_count / float(len(rows))) if rows else float("nan"),
        "rows": rows,
    }
    regression: Dict[str, Any] = {}
    if str(args.previous_summary_json).strip():
        prev_path = Path(args.previous_summary_json)
        if prev_path.exists():
            prev = json.loads(prev_path.read_text(encoding="utf-8-sig"))
            prev_rate = float(prev.get("pass_rate", float("nan")))
            curr_rate = float(summary.get("pass_rate", float("nan")))
            regression = {
                "previous_path": str(prev_path),
                "previous_pass_rate": prev_rate,
                "current_pass_rate": curr_rate,
                "delta_pass_rate": curr_rate - prev_rate if (not (prev_rate != prev_rate or curr_rate != curr_rate)) else float("nan"),
            }
            summary["regression"] = regression
    write_json(out_root / "campaign_summary.json", summary)
    lines: List[str] = ["# Parity Campaign Summary", ""]
    lines.append(f"- Pair count: `{summary['pair_count']}`")
    lines.append(f"- Pass count: `{summary['pass_count']}`")
    lines.append(f"- Pass rate: `{summary['pass_rate']:.4f}`")
    if regression:
        lines.append(
            f"- Previous pass rate: `{regression.get('previous_pass_rate', float('nan')):.4f}` "
            f"(delta `{regression.get('delta_pass_rate', float('nan')):.4f}`)"
        )
    lines.append("")
    lines.append("| Run | Pass | Equivalence Rate | Scenarios | Turns |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r.get('run_id')} | {r.get('pass')} | {float(r.get('equivalence_rate', float('nan'))):.4f} | "
            f"{int(r.get('compared_scenarios', 0))} | {int(r.get('compared_turns', 0))} |"
        )
    (out_root / "campaign_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"saved_summary={out_root / 'campaign_summary.json'}")
    min_pass_rate = max(0.0, min(1.0, float(args.require_min_pass_rate)))
    pass_rate = float(summary.get("pass_rate", float("nan")))
    if (not math.isnan(pass_rate)) and pass_rate < min_pass_rate:
        raise RuntimeError(
            f"Parity campaign pass_rate {pass_rate:.4f} below required threshold {min_pass_rate:.4f}."
        )
    if bool(args.fail_on_regression) and regression:
        delta = float(regression.get("delta_pass_rate", float("nan")))
        if not math.isnan(delta) and delta < 0.0:
            raise RuntimeError(f"Parity campaign regressed vs previous summary (delta={delta:.4f}).")


if __name__ == "__main__":
    main()

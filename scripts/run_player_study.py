#!/usr/bin/env python3
"""Player-study pipeline for NPC dialogue evaluation.

Subcommands:
- init: create participant-arm assignment sheet and questionnaire templates.
- report: ingest telemetry/questionnaire files, score SUS/immersion, export summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=list(fieldnames))
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def read_csv_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        out: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
        return out
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(dict(row))
    return rows


def parse_csv_list(raw: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for token in str(raw or "").replace(";", ",").split(","):
        t = token.strip()
        if not t:
            continue
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
    return out


def latin_square(arms: Sequence[str]) -> List[List[str]]:
    n = len(arms)
    if n == 0:
        return []
    base = list(arms)
    rows: List[List[str]] = []
    for i in range(n):
        rows.append(base[i:] + base[:i])
    return rows


def bootstrap_mean_ci(values: Sequence[float], seed: int = 42, n_boot: int = 1000) -> Dict[str, float]:
    cleaned = [float(v) for v in values if isinstance(v, (int, float)) and not math.isnan(float(v))]
    if not cleaned:
        return {"mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    rng = random.Random(seed)
    n = len(cleaned)
    means: List[float] = []
    for _ in range(max(200, int(n_boot))):
        sample = [cleaned[rng.randrange(n)] for _ in range(n)]
        means.append(float(sum(sample) / n))
    means.sort()
    return {
        "mean": float(sum(cleaned) / n),
        "ci95_low": float(means[int(0.025 * (len(means) - 1))]),
        "ci95_high": float(means[int(0.975 * (len(means) - 1))]),
    }


def to_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def sus_score(row: Dict[str, Any]) -> float:
    # SUS Q1..Q10 on 1..5 scale.
    vals: List[float] = []
    for i in range(1, 11):
        v = to_float(row.get(f"sus_q{i}", row.get(f"q{i}", float("nan"))))
        if math.isnan(v):
            return float("nan")
        vals.append(v)
    adjusted = 0.0
    for idx, v in enumerate(vals, start=1):
        if idx % 2 == 1:
            adjusted += (v - 1.0)
        else:
            adjusted += (5.0 - v)
    return adjusted * 2.5


def immersion_score(row: Dict[str, Any]) -> float:
    # Mean of immersion_q1..q5 if available.
    vals = []
    for i in range(1, 6):
        v = to_float(row.get(f"immersion_q{i}", row.get(f"im{i}", float("nan"))))
        if not math.isnan(v):
            vals.append(v)
    if not vals:
        return float("nan")
    return float(statistics.fmean(vals))


def cmd_init(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    arms = parse_csv_list(args.arms)
    if len(arms) < 2:
        raise ValueError("Need at least 2 arms for player study assignment.")
    participants = max(1, int(args.participants))
    square = latin_square(arms)
    rng = random.Random(int(args.seed))
    assignments: List[Dict[str, Any]] = []
    for idx in range(participants):
        pid = f"P{idx+1:03d}"
        arm_order = square[idx % len(square)][:]
        if bool(args.shuffle_within_row):
            rng.shuffle(arm_order)
        assignments.append(
            {
                "participant_id": pid,
                "arm_order": ",".join(arm_order),
                "notes": "",
            }
        )

    write_csv(
        out_dir / "participant_assignments.csv",
        assignments,
        fieldnames=["participant_id", "arm_order", "notes"],
    )

    telemetry_template = [
        {
            "participant_id": "P001",
            "arm_id": arms[0],
            "scenario_id": "s001",
            "task_success": 1,
            "turn_count": 4,
            "duration_ms": 42000,
            "dropout": 0,
        }
    ]
    write_csv(
        out_dir / "telemetry_template.csv",
        telemetry_template,
        fieldnames=["participant_id", "arm_id", "scenario_id", "task_success", "turn_count", "duration_ms", "dropout"],
    )

    questionnaire_template = [
        {
            "participant_id": "P001",
            "arm_id": arms[0],
            **{f"sus_q{i}": "" for i in range(1, 11)},
            **{f"immersion_q{i}": "" for i in range(1, 6)},
            "freeform_comment": "",
        }
    ]
    write_csv(
        out_dir / "questionnaire_template.csv",
        questionnaire_template,
        fieldnames=[
            "participant_id",
            "arm_id",
            *[f"sus_q{i}" for i in range(1, 11)],
            *[f"immersion_q{i}" for i in range(1, 6)],
            "freeform_comment",
        ],
    )

    write_json(
        out_dir / "study_manifest.json",
        {
            "participants": participants,
            "arms": arms,
            "seed": int(args.seed),
            "design": "balanced_latin_square",
        },
    )
    print(f"initialized_player_study={out_dir}")


def cmd_report(args: argparse.Namespace) -> None:
    telemetry_rows = read_csv_or_jsonl(Path(args.telemetry))
    questionnaire_rows = read_csv_or_jsonl(Path(args.questionnaire))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_arm: Dict[str, Dict[str, List[float]]] = {}

    for row in telemetry_rows:
        arm = str(row.get("arm_id", "unknown")).strip() or "unknown"
        payload = by_arm.setdefault(arm, {})
        payload.setdefault("task_success", []).append(to_float(row.get("task_success", float("nan"))))
        payload.setdefault("turn_count", []).append(to_float(row.get("turn_count", float("nan"))))
        payload.setdefault("duration_ms", []).append(to_float(row.get("duration_ms", float("nan"))))
        payload.setdefault("dropout", []).append(to_float(row.get("dropout", float("nan"))))

    enriched_q: List[Dict[str, Any]] = []
    for row in questionnaire_rows:
        out_row = dict(row)
        out_row["sus_score"] = sus_score(row)
        out_row["immersion_score"] = immersion_score(row)
        enriched_q.append(out_row)
        arm = str(row.get("arm_id", "unknown")).strip() or "unknown"
        payload = by_arm.setdefault(arm, {})
        payload.setdefault("sus_score", []).append(to_float(out_row["sus_score"]))
        payload.setdefault("immersion_score", []).append(to_float(out_row["immersion_score"]))

    summary_by_arm: Dict[str, Any] = {}
    for arm, metrics in sorted(by_arm.items()):
        metric_summary: Dict[str, Any] = {}
        for idx, (metric, vals) in enumerate(sorted(metrics.items())):
            metric_summary[metric] = bootstrap_mean_ci(vals, seed=int(args.seed) + idx * 13, n_boot=1200)
        summary_by_arm[arm] = metric_summary

    report = {
        "inputs": {
            "telemetry": str(args.telemetry),
            "questionnaire": str(args.questionnaire),
        },
        "row_counts": {
            "telemetry": len(telemetry_rows),
            "questionnaire": len(questionnaire_rows),
        },
        "arms": summary_by_arm,
    }
    write_json(out_dir / "player_study_summary.json", report)
    write_csv(
        out_dir / "questionnaire_scored.csv",
        enriched_q,
        fieldnames=list(enriched_q[0].keys()) if enriched_q else ["participant_id", "arm_id", "sus_score", "immersion_score"],
    )

    lines = ["# Player Study Summary", ""]
    lines.append(f"- Telemetry rows: `{len(telemetry_rows)}`")
    lines.append(f"- Questionnaire rows: `{len(questionnaire_rows)}`")
    lines.append("")
    lines.append("| Arm | SUS | Immersion | Task Success | Turn Count | Duration (ms) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for arm, metrics in summary_by_arm.items():
        sus = metrics.get("sus_score", {})
        imm = metrics.get("immersion_score", {})
        ts = metrics.get("task_success", {})
        tc = metrics.get("turn_count", {})
        dur = metrics.get("duration_ms", {})
        lines.append(
            f"| {arm} | {sus.get('mean', float('nan')):.2f} | {imm.get('mean', float('nan')):.2f} | "
            f"{ts.get('mean', float('nan')):.3f} | {tc.get('mean', float('nan')):.2f} | {dur.get('mean', float('nan')):.1f} |"
        )
    (out_dir / "player_study_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"saved_summary={out_dir / 'player_study_summary.json'}")
    print(f"saved_markdown={out_dir / 'player_study_summary.md'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init")
    p_init.add_argument("--participants", type=int, default=20)
    p_init.add_argument(
        "--arms",
        default="proposed_contextual_controlled,baseline_no_context,baseline_no_context_phi3_latest",
    )
    p_init.add_argument("--seed", type=int, default=42)
    p_init.add_argument("--shuffle-within-row", action="store_true")
    p_init.add_argument("--out-dir", default="storage/artifacts/player_study")
    p_init.set_defaults(func=cmd_init)

    p_report = sub.add_parser("report")
    p_report.add_argument("--telemetry", required=True, help="CSV/JSONL player telemetry.")
    p_report.add_argument("--questionnaire", required=True, help="CSV/JSONL questionnaire responses.")
    p_report.add_argument("--seed", type=int, default=42)
    p_report.add_argument("--out-dir", default="storage/artifacts/player_study")
    p_report.set_defaults(func=cmd_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()


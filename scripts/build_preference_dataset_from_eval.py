#!/usr/bin/env python3
"""Build a DPO/ORPO-ready preference dataset from proposal run artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_csv(raw: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for token in [x.strip() for x in str(raw).split(",") if x.strip()]:
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(token)
    return out


def choose_one_response_per_scenario(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    selected: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        sid = str(row.get("scenario_id", "")).strip()
        if not sid:
            continue
        key = (
            int(row.get("repeat_index", 999999) or 999999),
            int(row.get("request_index", 999999999) or 999999999),
        )
        prev = selected.get(sid)
        if prev is None:
            selected[sid] = dict(row)
            selected[sid]["_k"] = key
            continue
        if key < prev.get("_k", (999999, 999999999)):
            selected[sid] = dict(row)
            selected[sid]["_k"] = key
    for sid in list(selected.keys()):
        selected[sid].pop("_k", None)
    return selected


def read_human_eval_csv(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({k: (v if v is not None else "") for k, v in row.items()})
    return rows


def maybe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def aggregate_metric_by_scenario_arm(
    rows: List[Dict[str, Any]],
    metric: str,
) -> Dict[Tuple[str, str], Dict[str, float]]:
    buckets: Dict[Tuple[str, str], List[float]] = {}
    for row in rows:
        sid = str(row.get("scenario_id", "")).strip()
        arm = str(row.get("arm_id", "")).strip()
        if not sid or not arm:
            continue
        v = maybe_float(row.get(metric))
        if v != v:
            continue
        buckets.setdefault((sid, arm), []).append(v)

    out: Dict[Tuple[str, str], Dict[str, float]] = {}
    for key, vals in buckets.items():
        if not vals:
            continue
        out[key] = {
            "n": float(len(vals)),
            "mean": float(sum(vals) / len(vals)),
            "min": float(min(vals)),
            "max": float(max(vals)),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build preference dataset from proposal run + human-eval CSV.")
    parser.add_argument("--run-dir", required=True, help="Run directory under artifacts/proposal/<run_id>")
    parser.add_argument(
        "--human-eval-file",
        default="",
        help="Human eval CSV. Defaults to run_dir/human_eval_llm_multirater_consistent.csv if present, else human_eval_llm_multirater_large.csv.",
    )
    parser.add_argument("--target-arm", default="proposed_contextual_controlled")
    parser.add_argument("--baseline-arms", default="baseline_no_context,baseline_no_context_phi3_latest")
    parser.add_argument("--metric", default="overall_quality")
    parser.add_argument(
        "--score-scale-max",
        type=float,
        default=5.0,
        help="Max value for metric scale used in CSV (for normalized margin metadata).",
    )
    parser.add_argument("--min-margin", type=float, default=0.25, help="Minimum raw-score margin on selected metric.")
    parser.add_argument("--min-raters", type=int, default=2, help="Minimum rater count per arm per scenario.")
    parser.add_argument("--allow-reverse-pairs", action="store_true")
    parser.add_argument("--output-jsonl", default="", help="Output JSONL path")
    parser.add_argument("--output-summary", default="", help="Output summary JSON path")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    responses_dir = run_dir / "responses"
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")
    if not responses_dir.exists():
        raise FileNotFoundError(f"responses/ not found: {responses_dir}")

    if str(args.human_eval_file).strip():
        human_eval_path = Path(args.human_eval_file)
    else:
        p1 = run_dir / "human_eval_llm_multirater_consistent.csv"
        p2 = run_dir / "human_eval_llm_multirater_large.csv"
        human_eval_path = p1 if p1.exists() else p2
    if not human_eval_path.exists():
        raise FileNotFoundError(f"Human-eval CSV not found: {human_eval_path}")

    target_arm = str(args.target_arm).strip()
    baseline_arms = parse_csv(str(args.baseline_arms))
    if not target_arm:
        raise ValueError("target arm is required")
    if not baseline_arms:
        raise ValueError("at least one baseline arm is required")

    run_config = read_json(run_dir / "run_config.json")
    configured_arms = {
        str(item.get("arm_id", "")).strip()
        for item in run_config.get("arms", [])
        if str(item.get("arm_id", "")).strip()
    }
    if target_arm not in configured_arms:
        raise ValueError(f"target arm '{target_arm}' is not present in run_config arms")
    for arm in baseline_arms:
        if arm not in configured_arms:
            raise ValueError(f"baseline arm '{arm}' is not present in run_config arms")

    scenarios = {
        str(row.get("scenario_id", "")).strip(): row
        for row in read_jsonl(run_dir / "scenarios.jsonl")
        if str(row.get("scenario_id", "")).strip()
    }

    arm_rows: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for arm in [target_arm] + baseline_arms:
        path = responses_dir / f"{arm}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing response file: {path}")
        arm_rows[arm] = choose_one_response_per_scenario(read_jsonl(path))

    human_rows = read_human_eval_csv(human_eval_path)
    by_scenario_arm = aggregate_metric_by_scenario_arm(human_rows, metric=str(args.metric))

    output_rows: List[Dict[str, Any]] = []
    skipped_missing = 0
    skipped_raters = 0
    skipped_margin = 0

    for sid, scenario in scenarios.items():
        target_resp = arm_rows.get(target_arm, {}).get(sid)
        if target_resp is None:
            skipped_missing += 1
            continue

        target_score = by_scenario_arm.get((sid, target_arm))
        if target_score is None:
            skipped_missing += 1
            continue
        if int(target_score.get("n", 0)) < int(args.min_raters):
            skipped_raters += 1
            continue

        prompt = str(target_resp.get("prompt", "")).strip()
        chosen = str(target_resp.get("response", "")).strip()
        if not prompt or not chosen:
            skipped_missing += 1
            continue

        for baseline_arm in baseline_arms:
            base_resp = arm_rows.get(baseline_arm, {}).get(sid)
            base_score = by_scenario_arm.get((sid, baseline_arm))
            if base_resp is None or base_score is None:
                skipped_missing += 1
                continue
            if int(base_score.get("n", 0)) < int(args.min_raters):
                skipped_raters += 1
                continue

            rejected = str(base_resp.get("response", "")).strip()
            if not rejected:
                skipped_missing += 1
                continue

            delta = float(target_score.get("mean", 0.0)) - float(base_score.get("mean", 0.0))
            margin_ok = delta >= float(args.min_margin)
            reverse_ok = bool(args.allow_reverse_pairs) and (-delta) >= float(args.min_margin)
            if not margin_ok and not reverse_ok:
                skipped_margin += 1
                continue

            if margin_ok:
                row = {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "metadata": {
                        "scenario_id": sid,
                        "target_arm": target_arm,
                        "baseline_arm": baseline_arm,
                        "preference_direction": "target_over_baseline",
                        "metric": str(args.metric),
                        "target_score_mean": float(target_score.get("mean", 0.0)),
                        "baseline_score_mean": float(base_score.get("mean", 0.0)),
                        "score_delta_raw": delta,
                        "score_delta_norm": (delta / float(args.score_scale_max)) if float(args.score_scale_max) > 0 else 0.0,
                        "target_raters": int(target_score.get("n", 0)),
                        "baseline_raters": int(base_score.get("n", 0)),
                        "persona": str(scenario.get("persona", "")),
                        "player_input": str(scenario.get("player_input", "")),
                    },
                }
                output_rows.append(row)

            if reverse_ok:
                row = {
                    "prompt": prompt,
                    "chosen": rejected,
                    "rejected": chosen,
                    "metadata": {
                        "scenario_id": sid,
                        "target_arm": target_arm,
                        "baseline_arm": baseline_arm,
                        "preference_direction": "baseline_over_target",
                        "metric": str(args.metric),
                        "target_score_mean": float(target_score.get("mean", 0.0)),
                        "baseline_score_mean": float(base_score.get("mean", 0.0)),
                        "score_delta_raw": delta,
                        "score_delta_norm": (delta / float(args.score_scale_max)) if float(args.score_scale_max) > 0 else 0.0,
                        "target_raters": int(target_score.get("n", 0)),
                        "baseline_raters": int(base_score.get("n", 0)),
                        "persona": str(scenario.get("persona", "")),
                        "player_input": str(scenario.get("player_input", "")),
                    },
                }
                output_rows.append(row)

    output_jsonl = (
        Path(args.output_jsonl)
        if str(args.output_jsonl).strip()
        else run_dir / "preference_dataset.jsonl"
    )
    output_summary = (
        Path(args.output_summary)
        if str(args.output_summary).strip()
        else run_dir / "preference_dataset_summary.json"
    )

    write_jsonl(output_jsonl, output_rows)

    deltas = [
        float(row.get("metadata", {}).get("score_delta_raw", 0.0))
        for row in output_rows
        if isinstance(row.get("metadata"), dict)
    ]
    summary = {
        "run_dir": str(run_dir),
        "human_eval_file": str(human_eval_path),
        "target_arm": target_arm,
        "baseline_arms": baseline_arms,
        "metric": str(args.metric),
        "row_count": len(output_rows),
        "skipped": {
            "missing": skipped_missing,
            "insufficient_raters": skipped_raters,
            "insufficient_margin": skipped_margin,
        },
        "score_delta_raw": {
            "mean": (sum(deltas) / len(deltas)) if deltas else float("nan"),
            "min": min(deltas) if deltas else float("nan"),
            "max": max(deltas) if deltas else float("nan"),
        },
        "output_jsonl": str(output_jsonl),
    }
    write_json(output_summary, summary)

    print(f"Preference dataset: {output_jsonl}")
    print(f"Rows: {len(output_rows)}")
    print(f"Summary: {output_summary}")


if __name__ == "__main__":
    main()


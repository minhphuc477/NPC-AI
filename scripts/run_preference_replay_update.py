#!/usr/bin/env python3
"""Build weak preference replay dataset from scored runs + implicit feedback.

This is a safe adapter-only replay helper:
1) mine weak preference pairs from scored JSONL rows (same scenario, different arm responses),
2) optionally blend implicit feedback priors,
3) optionally launch a short DPO adapter update.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


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
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_score(row: Dict[str, Any]) -> float:
    explicit = row.get("overall_quality")
    if isinstance(explicit, (int, float)) and math.isfinite(float(explicit)):
        return float(explicit)
    context = float(row.get("response_context_coverage", 0.0) or 0.0)
    persona = float(row.get("response_persona_coverage", 0.0) or 0.0)
    natural = float(row.get("naturalness", 0.5) or 0.5)
    penalty = 0.0
    if bool(row.get("response_fallback")) or str(row.get("response_control_source", "")) == "fallback":
        penalty += 0.2
    if not bool(row.get("ok", True)):
        penalty += 0.2
    return max(0.0, min(1.0, 0.45 * context + 0.35 * persona + 0.20 * natural - penalty))


def load_feedback(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"count": 0, "mean_score": float("nan"), "continued_rate": float("nan")}
    rows = read_jsonl(path)
    if not rows:
        return {"count": 0, "mean_score": float("nan"), "continued_rate": float("nan")}
    scores = [float(r.get("score", 0.0) or 0.0) for r in rows if isinstance(r.get("score", 0.0), (int, float))]
    cont = [1.0 if str(r.get("outcome", "")).strip().lower() == "continued" else 0.0 for r in rows]
    mean_score = float(sum(scores) / len(scores)) if scores else float("nan")
    continued_rate = float(sum(cont) / len(cont)) if cont else float("nan")
    return {"count": len(rows), "mean_score": mean_score, "continued_rate": continued_rate}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scores-glob",
        default="storage/artifacts/proposal/*/scores/*.jsonl",
        help="Glob for scored rows containing prompt/response/overall_quality.",
    )
    parser.add_argument(
        "--feedback-jsonl",
        default="storage/artifacts/feedback/implicit_feedback.jsonl",
        help="Implicit feedback JSONL.",
    )
    parser.add_argument("--min-score-gap", type=float, default=0.10)
    parser.add_argument("--max-pairs", type=int, default=5000)
    parser.add_argument("--out-jsonl", default="storage/artifacts/feedback/replay_preference_pairs.jsonl")
    parser.add_argument("--out-summary", default="storage/artifacts/feedback/replay_preference_summary.json")
    parser.add_argument("--run-dpo-ministep", action="store_true")
    parser.add_argument("--base-model", default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--dpo-output-dir", default="storage/outputs/dpo_replay_update")
    parser.add_argument("--dpo-max-steps", type=int, default=80)
    parser.add_argument("--dpo-batch-size", type=int, default=1)
    parser.add_argument("--dpo-grad-acc", type=int, default=8)
    args = parser.parse_args()

    score_paths = [Path(p) for p in sorted(glob.glob(str(args.scores_glob)))]
    all_rows: List[Dict[str, Any]] = []
    for path in score_paths:
        all_rows.extend(read_jsonl(path))

    by_case: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        sid = str(row.get("scenario_id", "")).strip()
        rep = str(row.get("repeat_index", "")).strip()
        key = f"{sid}|{rep}" if sid else ""
        if not key:
            continue
        if not str(row.get("response", "")).strip():
            continue
        by_case[key].append(row)

    feedback_stats = load_feedback(Path(args.feedback_jsonl))
    feedback_weight = 1.0
    if isinstance(feedback_stats.get("continued_rate"), float) and math.isfinite(float(feedback_stats["continued_rate"])):
        # Conservative prior: low continuation slightly increases minimum score gap.
        feedback_weight = max(0.75, min(1.25, 1.0 + (float(feedback_stats["continued_rate"]) - 0.5)))

    effective_gap = max(0.01, float(args.min_score_gap) / feedback_weight)
    pairs: List[Dict[str, Any]] = []
    for case_id, rows in by_case.items():
        if len(rows) < 2:
            continue
        scored: List[Tuple[float, Dict[str, Any]]] = sorted(
            [(safe_score(r), r) for r in rows],
            key=lambda item: item[0],
            reverse=True,
        )
        best_score, best_row = scored[0]
        worst_score, worst_row = scored[-1]
        gap = float(best_score - worst_score)
        if gap < effective_gap:
            continue
        prompt = str(best_row.get("prompt", "")).strip() or str(worst_row.get("prompt", "")).strip()
        chosen = str(best_row.get("response", "")).strip()
        rejected = str(worst_row.get("response", "")).strip()
        if not prompt or not chosen or not rejected or chosen == rejected:
            continue
        sid, rep = case_id.split("|", 1)
        pairs.append(
            {
                "id": f"replay_{sid}_{rep}_{len(pairs):06d}",
                "source": "replay_weak_preference",
                "scenario_id": sid,
                "repeat_index": rep,
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "chosen_arm": best_row.get("arm_id", ""),
                "rejected_arm": worst_row.get("arm_id", ""),
                "chosen_score": best_score,
                "rejected_score": worst_score,
                "score_gap": gap,
            }
        )
        if len(pairs) >= int(args.max_pairs):
            break

    out_jsonl = Path(args.out_jsonl)
    out_summary = Path(args.out_summary)
    write_jsonl(out_jsonl, pairs)
    summary = {
        "score_files": [str(p) for p in score_paths],
        "score_file_count": len(score_paths),
        "rows_loaded": len(all_rows),
        "cases_considered": len(by_case),
        "pairs_built": len(pairs),
        "min_score_gap": float(args.min_score_gap),
        "effective_min_score_gap": effective_gap,
        "feedback_stats": feedback_stats,
        "sample_pair_ids": [p["id"] for p in pairs[:10]],
    }
    write_json(out_summary, summary)
    print(f"saved_pairs={out_jsonl}")
    print(f"saved_summary={out_summary}")

    if bool(args.run_dpo_ministep):
        if not pairs:
            raise RuntimeError("No replay pairs built; aborting DPO ministep.")
        cmd = [
            sys.executable,
            "scripts/train_dpo.py",
            "--dataset",
            str(out_jsonl),
            "--base-model",
            str(args.base_model),
            "--output-dir",
            str(args.dpo_output_dir),
            "--max-steps",
            str(int(args.dpo_max_steps)),
            "--batch-size",
            str(int(args.dpo_batch_size)),
            "--grad-acc",
            str(int(args.dpo_grad_acc)),
            "--epochs",
            "1",
            "--logging-steps",
            "5",
            "--save-steps",
            "20",
        ]
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"DPO ministep failed with code {proc.returncode}")


if __name__ == "__main__":
    main()

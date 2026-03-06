#!/usr/bin/env python3
"""Build a blind, multi-rater human-eval package from a proposal artifact run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from statistics import NormalDist
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


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
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def parse_csv_arg(raw: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for token in (x.strip() for x in raw.split(",")):
        if not token:
            continue
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(token)
    return out


def choose_single_response_per_scenario(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    selected: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        sid = str(row.get("scenario_id", "")).strip()
        if not sid:
            continue
        current = selected.get(sid)
        if current is None:
            selected[sid] = dict(row)
            continue
        # Prefer repeat_index=0 and lower request_index for deterministic selection.
        cur_repeat = int(current.get("repeat_index", 999999) or 999999)
        new_repeat = int(row.get("repeat_index", 999999) or 999999)
        if new_repeat < cur_repeat:
            selected[sid] = dict(row)
            continue
        if new_repeat == cur_repeat:
            cur_req = int(current.get("request_index", 999999999) or 999999999)
            new_req = int(row.get("request_index", 999999999) or 999999999)
            if new_req < cur_req:
                selected[sid] = dict(row)
    return selected


def split_scenarios_for_annotators(
    scenario_ids: Sequence[str],
    annotators: Sequence[str],
    shared_ratio: float,
    ratings_per_scenario: int,
    seed: int,
) -> Tuple[List[str], Dict[str, List[str]], Dict[str, List[str]]]:
    ids = list(scenario_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    if not ids:
        return [], {a: [] for a in annotators}, {}

    ratio = max(0.0, min(1.0, float(shared_ratio)))
    shared_count = int(round(len(ids) * ratio))
    shared = ids[:shared_count]
    remaining = ids[shared_count:]

    assigned: Dict[str, List[str]] = {a: [] for a in annotators}
    scenario_to_annotators: Dict[str, List[str]] = {}
    if not annotators:
        return shared, assigned, scenario_to_annotators

    ann_order = list(annotators)
    rng.shuffle(ann_order)
    k = max(1, min(len(annotators), int(ratings_per_scenario)))

    # Shared subset: all annotators rate the same scenarios.
    for sid in shared:
        scenario_to_annotators[sid] = list(annotators)
        for ann in annotators:
            assigned[ann].append(sid)

    # Non-shared subset: each scenario is rated by k annotators using rotating assignments.
    cursor = 0
    for sid in remaining:
        if cursor > 0 and (cursor % len(ann_order) == 0):
            rng.shuffle(ann_order)
        start = cursor % len(ann_order)
        chosen = [ann_order[(start + j) % len(ann_order)] for j in range(k)]
        scenario_to_annotators[sid] = chosen
        for ann in chosen:
            assigned[ann].append(sid)
        cursor += 1

    return shared, assigned, scenario_to_annotators


def recommended_paired_samples(
    mde: float,
    std_dev: float,
    alpha: float,
    power: float,
    design_effect: float,
) -> int:
    delta = max(1e-6, float(mde))
    sigma = max(1e-6, float(std_dev))
    a = min(max(float(alpha), 1e-6), 0.5)
    p = min(max(float(power), 0.50), 0.999)
    deff = max(1.0, float(design_effect))

    nd = NormalDist()
    z_alpha = nd.inv_cdf(1.0 - a / 2.0)
    z_beta = nd.inv_cdf(p)
    n = ((z_alpha + z_beta) * sigma / delta) ** 2
    return int(math.ceil(n * deff))


def overlap_matrix(assigned: Dict[str, List[str]], annotators: Sequence[str]) -> Dict[str, Dict[str, int]]:
    by_annotator = {ann: set(assigned.get(ann, [])) for ann in annotators}
    out: Dict[str, Dict[str, int]] = {}
    for a in annotators:
        row: Dict[str, int] = {}
        for b in annotators:
            row[b] = len(by_annotator.get(a, set()).intersection(by_annotator.get(b, set())))
        out[a] = row
    return out


def iter_rows_for_annotator(
    annotator_id: str,
    scenario_ids: Iterable[str],
    scenarios_by_id: Dict[str, Dict[str, Any]],
    responses_by_arm_and_sid: Dict[str, Dict[str, Dict[str, Any]]],
    arm_blind_map: Dict[str, str],
) -> Iterable[Dict[str, Any]]:
    for sid in scenario_ids:
        scenario = scenarios_by_id.get(sid, {})
        for arm_id, by_sid in responses_by_arm_and_sid.items():
            response_row = by_sid.get(sid, {})
            yield {
                "scenario_id": sid,
                "arm_id": arm_id,
                "arm_blind_id": arm_blind_map.get(arm_id, arm_id),
                "annotator_id": annotator_id,
                "persona": str(scenario.get("persona", "")),
                "dynamic_context": str(scenario.get("dynamic_context", "")),
                "player_input": str(scenario.get("player_input", "")),
                "response": str(response_row.get("response", "")),
                "context_relevance": "",
                "persona_consistency": "",
                "naturalness": "",
                "overall_quality": "",
                "notes": "",
            }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build blind human-eval annotation pack from proposal run artifacts.")
    parser.add_argument("--run-dir", required=True, help="Artifact run dir (e.g., artifacts/proposal/<run_id>)")
    parser.add_argument(
        "--arms",
        default="proposed_contextual_controlled,proposed_contextual,candidate_no_context,baseline_no_context,baseline_no_context_phi3_latest",
        help="Comma-separated arm IDs to include if present.",
    )
    parser.add_argument(
        "--annotators",
        default="annotator_1,annotator_2,annotator_3",
        help="Comma-separated annotator IDs",
    )
    parser.add_argument("--shared-ratio", type=float, default=0.25, help="Fraction of scenarios shared across all annotators")
    parser.add_argument(
        "--ratings-per-scenario",
        type=int,
        default=2,
        help="How many annotators rate each non-shared scenario (capped by annotator count).",
    )
    parser.add_argument("--max-scenarios", type=int, default=0, help="Optional scenario cap (0 = all)")
    parser.add_argument(
        "--target-power",
        type=float,
        default=0.80,
        help="Target statistical power for paired human-preference comparisons (normal approximation).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Two-sided alpha for planning paired human-study sample size.",
    )
    parser.add_argument(
        "--mde",
        type=float,
        default=0.08,
        help="Minimum detectable effect size on normalized [0,1] scale for planning.",
    )
    parser.add_argument(
        "--std-dev",
        type=float,
        default=0.25,
        help="Assumed standard deviation of paired scenario-level differences on [0,1] scale.",
    )
    parser.add_argument(
        "--design-effect",
        type=float,
        default=1.2,
        help="Design effect inflation factor for clustered/multi-rater sampling.",
    )
    parser.add_argument("--seed", type=int, default=19, help="Random seed")
    parser.add_argument("--output-dir", default="data/human_eval_pack", help="Output directory root")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")
    responses_dir = run_dir / "responses"
    if not responses_dir.exists():
        raise FileNotFoundError(f"responses/ not found under run dir: {responses_dir}")

    scenarios_path = run_dir / "scenarios.jsonl"
    if not scenarios_path.exists():
        raise FileNotFoundError(f"scenarios.jsonl not found: {scenarios_path}")
    scenarios = read_jsonl(scenarios_path)
    scenarios_by_id = {str(r.get("scenario_id", "")).strip(): r for r in scenarios if str(r.get("scenario_id", "")).strip()}

    requested_arms = parse_csv_arg(str(args.arms))
    present_arms: List[str] = []
    responses_by_arm_and_sid: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for arm in requested_arms:
        path = responses_dir / f"{arm}.jsonl"
        if not path.exists():
            continue
        rows = read_jsonl(path)
        picked = choose_single_response_per_scenario(rows)
        if not picked:
            continue
        present_arms.append(arm)
        responses_by_arm_and_sid[arm] = picked
    if not present_arms:
        raise RuntimeError("No requested arm response files found for this run.")

    scenario_ids = sorted(
        sid
        for sid in scenarios_by_id.keys()
        if all(sid in responses_by_arm_and_sid[arm] for arm in present_arms)
    )
    if not scenario_ids:
        raise RuntimeError("No scenario IDs have responses for all selected arms.")

    annotators = parse_csv_arg(str(args.annotators))
    if not annotators:
        raise RuntimeError("At least one annotator is required.")
    ratings_per_scenario = max(1, min(len(annotators), int(args.ratings_per_scenario)))

    planned_pairs = recommended_paired_samples(
        mde=float(args.mde),
        std_dev=float(args.std_dev),
        alpha=float(args.alpha),
        power=float(args.target_power),
        design_effect=float(args.design_effect),
    )
    recommended_scenarios = int(math.ceil(planned_pairs / float(ratings_per_scenario)))

    scenario_pool = list(scenario_ids)
    rng_select = random.Random(args.seed + 31)
    rng_select.shuffle(scenario_pool)
    if int(args.max_scenarios) == 0:
        selected_scenario_ids = list(scenario_pool)
        if recommended_scenarios > len(selected_scenario_ids):
            selected_scenario_ids = list(scenario_pool[: min(recommended_scenarios, len(scenario_pool))])
    else:
        selected_scenario_ids = list(scenario_pool[: int(args.max_scenarios)])
    selected_scenario_ids = sorted(selected_scenario_ids)

    shared_ids, assigned_ids, scenario_to_annotators = split_scenarios_for_annotators(
        scenario_ids=selected_scenario_ids,
        annotators=annotators,
        shared_ratio=float(args.shared_ratio),
        ratings_per_scenario=ratings_per_scenario,
        seed=args.seed,
    )

    arm_blind_map: Dict[str, str] = {}
    rng_blind = random.Random(args.seed + 97)
    shuffled_arms = list(present_arms)
    rng_blind.shuffle(shuffled_arms)
    for idx, arm in enumerate(shuffled_arms):
        arm_blind_map[arm] = f"Arm_{chr(ord('A') + idx)}"

    out_root = Path(args.output_dir) / run_dir.name
    out_root.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "scenario_id",
        "arm_id",
        "arm_blind_id",
        "annotator_id",
        "persona",
        "dynamic_context",
        "player_input",
        "response",
        "context_relevance",
        "persona_consistency",
        "naturalness",
        "overall_quality",
        "notes",
    ]

    total_rows = 0
    for annotator in annotators:
        scenario_set = sorted(set(shared_ids + assigned_ids.get(annotator, [])))
        out_path = out_root / f"{annotator}.csv"
        with out_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in iter_rows_for_annotator(
                annotator_id=annotator,
                scenario_ids=scenario_set,
                scenarios_by_id=scenarios_by_id,
                responses_by_arm_and_sid=responses_by_arm_and_sid,
                arm_blind_map=arm_blind_map,
            ):
                writer.writerow(row)
                total_rows += 1

    merged_path = out_root / "ratings_merged_template.csv"
    with merged_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

    write_json(
        out_root / "blind_mapping.json",
        {
            "run_dir": str(run_dir),
            "present_arms": present_arms,
            "arm_blind_map": arm_blind_map,
        },
    )
    write_json(
        out_root / "assignment_plan.json",
        {
            "run_dir": str(run_dir),
            "scenario_count": len(selected_scenario_ids),
            "shared_scenarios": shared_ids,
            "annotators": annotators,
            "assigned_scenarios": assigned_ids,
            "scenario_to_annotators": scenario_to_annotators,
            "ratings_per_scenario": ratings_per_scenario,
            "overlap_matrix": overlap_matrix(assigned_ids, annotators),
            "power_plan": {
                "target_power": float(args.target_power),
                "alpha": float(args.alpha),
                "mde": float(args.mde),
                "std_dev": float(args.std_dev),
                "design_effect": float(args.design_effect),
                "planned_paired_samples": planned_pairs,
                "recommended_scenarios": recommended_scenarios,
                "selected_scenarios": len(selected_scenario_ids),
                "scenario_shortfall": max(0, recommended_scenarios - len(selected_scenario_ids)),
            },
            "rows_per_annotator_expected": {
                ann: len(set(shared_ids + assigned_ids.get(ann, []))) * len(present_arms)
                for ann in annotators
            },
            "total_rows_expected": total_rows,
        },
    )

    readme = out_root / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Human Eval Pack",
                "",
                "## Files",
                "- `<annotator>.csv`: per-rater blind worksheet.",
                "- `blind_mapping.json`: hidden mapping from blind IDs to true arms.",
                "- `assignment_plan.json`: scenario split and expected row counts.",
                "- `ratings_merged_template.csv`: merge target file.",
                "",
                "## Study Design Notes",
                f"- Ratings per non-shared scenario: `{ratings_per_scenario}`.",
                "- `assignment_plan.json` contains overlap matrix and power-based sample-size planning.",
                "",
                "## Workflow",
                "1. Send each annotator only their own CSV.",
                "2. Collect completed files and concatenate rows into `ratings_merged_template.csv`.",
                "3. Run proposal evaluation with:",
                "```bash",
                f"python scripts/run_proposal_alignment_eval_batched.py --scenarios {scenarios_path} --human-eval-file {merged_path}",
                "```",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Human-eval pack generated: {out_root}")
    print(f"Arms: {', '.join(present_arms)}")
    print(f"Scenarios: {len(selected_scenario_ids)} (shared={len(shared_ids)})")
    print(f"Total annotator rows: {total_rows}")
    print(f"Power plan: paired_samples={planned_pairs}, recommended_scenarios={recommended_scenarios}")


if __name__ == "__main__":
    main()

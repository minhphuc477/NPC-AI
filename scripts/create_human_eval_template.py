#!/usr/bin/env python3
"""Create a CSV template for human evaluation ratings."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def parse_csv_arg(raw: str) -> List[str]:
    out: List[str] = []
    for item in raw.split(","):
        token = item.strip()
        if token:
            out.append(token)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Create human-eval rating template CSV.")
    parser.add_argument("--scenarios", default="data/proposal_eval_scenarios_large.jsonl", help="Scenario JSONL path")
    parser.add_argument(
        "--arms",
        default="proposed_contextual_controlled,proposed_contextual,candidate_no_context,baseline_no_context",
        help="Comma-separated arm IDs",
    )
    parser.add_argument("--annotators", default="annotator_1,annotator_2", help="Comma-separated annotator IDs")
    parser.add_argument("--output", default="data/proposal_human_eval_template.csv", help="Output CSV path")
    args = parser.parse_args()

    scenario_rows = read_jsonl(Path(args.scenarios))
    arm_ids = parse_csv_arg(args.arms)
    annotators = parse_csv_arg(args.annotators)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scenario_id",
                "arm_id",
                "annotator_id",
                "context_relevance",
                "persona_consistency",
                "naturalness",
                "overall_quality",
                "notes",
            ],
        )
        writer.writeheader()
        for scenario in scenario_rows:
            sid = str(scenario.get("scenario_id", "")).strip()
            if not sid:
                continue
            for arm_id in arm_ids:
                for annotator in annotators:
                    writer.writerow(
                        {
                            "scenario_id": sid,
                            "arm_id": arm_id,
                            "annotator_id": annotator,
                            "context_relevance": "",
                            "persona_consistency": "",
                            "naturalness": "",
                            "overall_quality": "",
                            "notes": "",
                        }
                    )

    print(f"Template written: {out_path}")
    print(f"Rows: {len(scenario_rows) * len(arm_ids) * len(annotators)}")


if __name__ == "__main__":
    main()

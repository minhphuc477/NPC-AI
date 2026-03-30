#!/usr/bin/env python3
"""Run parity verification between harness and UE5 response traces.

Protocol criteria (defaults):
- At least 20 unique scenarios compared.
- At least 98% turn-level output equivalence.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def normalize_text(text: str) -> str:
    text = str(text or "")
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s']", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def token_jaccard(a: str, b: str) -> float:
    sa = set(normalize_text(a).split())
    sb = set(normalize_text(b).split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return float(len(sa & sb) / len(sa | sb))


def _get_intish(row: Dict[str, Any], keys: Sequence[str]) -> int:
    for key in keys:
        value = row.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
    return 0


def make_key(row: Dict[str, Any]) -> Tuple[str, int, int]:
    scenario_id = str(row.get("scenario_id", "")).strip() or "unknown"
    repeat_index = _get_intish(row, ["repeat_index", "repeat", "run_repeat"])
    turn_index = _get_intish(row, ["request_index", "turn_index", "turn_id"])
    return (scenario_id, repeat_index, turn_index)


@dataclass
class MatchRow:
    key: Tuple[str, int, int]
    harness_text: str
    ue5_text: str
    exact: bool
    near_match: bool
    jaccard: float


def build_map(rows: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, int, int], Dict[str, Any]]:
    out: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = make_key(row)
        out[key] = row
    return out


def render_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Parity Verification Report")
    lines.append("")
    lines.append(f"- Harness file: `{report['inputs']['harness_jsonl']}`")
    lines.append(f"- UE5 file: `{report['inputs']['ue5_jsonl']}`")
    lines.append(f"- Compared turns: `{report['counts']['compared_turns']}`")
    lines.append(f"- Compared scenarios: `{report['counts']['compared_scenarios']}`")
    lines.append(f"- Equivalence rate: `{report['metrics']['equivalence_rate']:.4f}`")
    lines.append(f"- Exact match rate: `{report['metrics']['exact_match_rate']:.4f}`")
    lines.append(f"- Near-match rate: `{report['metrics']['near_match_rate']:.4f}`")
    lines.append(f"- Protocol pass: `{report['pass']}`")
    lines.append("")
    lines.append("## Thresholds")
    lines.append(f"- Min scenarios: `{report['thresholds']['min_scenarios']}`")
    lines.append(f"- Min equivalence rate: `{report['thresholds']['min_equivalence_rate']}`")
    lines.append(f"- Jaccard near-match threshold: `{report['thresholds']['near_match_jaccard']}`")
    lines.append("")
    if report.get("notes"):
        lines.append("## Notes")
        for note in report["notes"]:
            lines.append(f"- {note}")
        lines.append("")
    mismatches = report.get("mismatches", [])
    if mismatches:
        lines.append("## Sample Mismatches")
        for row in mismatches[:20]:
            lines.append(
                f"- `{row['scenario_id']}|r{row['repeat_index']}|t{row['turn_index']}` "
                f"(jaccard={row['jaccard']:.4f})"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--harness-jsonl", required=True)
    parser.add_argument("--ue5-jsonl", required=True)
    parser.add_argument("--min-scenarios", type=int, default=20)
    parser.add_argument("--min-equivalence-rate", type=float, default=0.98)
    parser.add_argument("--near-match-jaccard", type=float, default=0.92)
    parser.add_argument("--output-json", default="storage/artifacts/publication_profiles/parity_verification.json")
    parser.add_argument("--output-md", default="storage/artifacts/publication_profiles/parity_verification.md")
    args = parser.parse_args()

    harness_path = Path(args.harness_jsonl)
    ue5_path = Path(args.ue5_jsonl)
    harness_rows = read_jsonl(harness_path)
    ue5_rows = read_jsonl(ue5_path)

    harness_map = build_map(harness_rows)
    ue5_map = build_map(ue5_rows)
    common_keys = sorted(set(harness_map.keys()) & set(ue5_map.keys()))

    matches: List[MatchRow] = []
    for key in common_keys:
        h = harness_map[key]
        u = ue5_map[key]
        h_text = str(h.get("response", h.get("text", "")))
        u_text = str(u.get("response", u.get("text", "")))
        h_norm = normalize_text(h_text)
        u_norm = normalize_text(u_text)
        jac = token_jaccard(h_norm, u_norm)
        exact = h_norm == u_norm
        near = jac >= float(args.near_match_jaccard)
        matches.append(
            MatchRow(
                key=key,
                harness_text=h_text,
                ue5_text=u_text,
                exact=exact,
                near_match=near,
                jaccard=jac,
            )
        )

    compared_turns = len(matches)
    compared_scenarios = len({m.key[0] for m in matches})
    exact_count = sum(1 for m in matches if m.exact)
    near_count = sum(1 for m in matches if (not m.exact) and m.near_match)
    equivalent_count = sum(1 for m in matches if m.exact or m.near_match)

    exact_rate = float(exact_count / compared_turns) if compared_turns else 0.0
    near_rate = float(near_count / compared_turns) if compared_turns else 0.0
    equivalence_rate = float(equivalent_count / compared_turns) if compared_turns else 0.0

    notes: List[str] = []
    if compared_turns == 0:
        notes.append("No overlapping keys between harness and UE5 files.")
    if compared_scenarios < int(args.min_scenarios):
        notes.append(
            f"Compared scenarios below threshold ({compared_scenarios} < {int(args.min_scenarios)})."
        )
    if equivalence_rate < float(args.min_equivalence_rate):
        notes.append(
            f"Equivalence rate below threshold ({equivalence_rate:.4f} < {float(args.min_equivalence_rate):.4f})."
        )

    protocol_pass = (
        compared_scenarios >= int(args.min_scenarios)
        and equivalence_rate >= float(args.min_equivalence_rate)
        and compared_turns > 0
    )

    mismatches: List[Dict[str, Any]] = []
    for m in matches:
        if m.exact or m.near_match:
            continue
        mismatches.append(
            {
                "scenario_id": m.key[0],
                "repeat_index": m.key[1],
                "turn_index": m.key[2],
                "jaccard": m.jaccard,
                "harness_response": m.harness_text,
                "ue5_response": m.ue5_text,
            }
        )
    mismatches.sort(key=lambda x: x["jaccard"])

    report: Dict[str, Any] = {
        "pass": protocol_pass,
        "inputs": {
            "harness_jsonl": str(harness_path),
            "ue5_jsonl": str(ue5_path),
        },
        "thresholds": {
            "min_scenarios": int(args.min_scenarios),
            "min_equivalence_rate": float(args.min_equivalence_rate),
            "near_match_jaccard": float(args.near_match_jaccard),
        },
        "counts": {
            "harness_turns": len(harness_rows),
            "ue5_turns": len(ue5_rows),
            "compared_turns": compared_turns,
            "compared_scenarios": compared_scenarios,
        },
        "metrics": {
            "exact_match_rate": exact_rate,
            "near_match_rate": near_rate,
            "equivalence_rate": equivalence_rate,
        },
        "notes": notes,
        "mismatches": mismatches[:100],
    }

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    write_json(out_json, report)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_markdown(report), encoding="utf-8")
    print(f"Parity verification json: {out_json}")
    print(f"Parity verification md: {out_md}")
    print(f"pass={protocol_pass}")


if __name__ == "__main__":
    main()


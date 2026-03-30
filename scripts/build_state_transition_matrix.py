#!/usr/bin/env python3
"""Build behavior-state transition matrix for SAGE prefetch.

Input supports either:
  1) scenario JSONL with `turns:[{behavior_state:...}, ...]`
  2) flat JSONL rows with `scenario_id` + `turn_index` + `behavior_state`
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


DEFAULT_STATE_ORDER: List[str] = [
    "patrolling",
    "negotiating",
    "detained",
    "assisting",
    "observing",
    "researching",
    "investigating",
    "crafting",
    "guarding",
    "quest_handoff",
    "recovery",
    "combat_ready",
]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def build_counts_nested_turns(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        turns = row.get("turns")
        if not isinstance(turns, list) or len(turns) < 2:
            continue
        seq: List[str] = []
        for t in turns:
            if isinstance(t, dict):
                s = str(t.get("behavior_state", "")).strip().lower()
                if s:
                    seq.append(s)
        for i in range(len(seq) - 1):
            counts[seq[i]][seq[i + 1]] += 1
    return counts


def build_counts_flat(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_scenario: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    for row in rows:
        sid = str(row.get("scenario_id", "")).strip()
        if not sid:
            continue
        s = str(row.get("behavior_state", "")).strip().lower()
        if not s:
            continue
        try:
            idx = int(row.get("turn_index", 0))
        except Exception:
            idx = 0
        by_scenario[sid].append((idx, s))
    for seq_rows in by_scenario.values():
        seq_rows.sort(key=lambda x: x[0])
        seq = [s for _, s in seq_rows]
        for i in range(len(seq) - 1):
            counts[seq[i]][seq[i + 1]] += 1
    return counts


def normalize_counts(counts: Dict[str, Dict[str, int]], state_order: List[str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for src in state_order:
        row_counts = counts.get(src, {})
        total = float(sum(row_counts.values()))
        if total <= 0:
            # fallback to self-loop if unseen
            out[src] = {dst: (1.0 if dst == src else 0.0) for dst in state_order}
            continue
        out[src] = {dst: float(row_counts.get(dst, 0)) / total for dst in state_order}
    return out


def row_major(matrix: Dict[str, Dict[str, float]], state_order: List[str]) -> List[float]:
    vals: List[float] = []
    for src in state_order:
        for dst in state_order:
            vals.append(float(matrix.get(src, {}).get(dst, 0.0)))
    return vals


def top_next(matrix: Dict[str, Dict[str, float]], top_k: int = 3) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for src, row in matrix.items():
        ranked = sorted(row.items(), key=lambda x: x[1], reverse=True)
        out[src] = [k for k, v in ranked[:top_k] if v > 0.0]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SAGE state transition matrix.")
    parser.add_argument("--input", default="data/benchmark_scenarios.jsonl")
    parser.add_argument("--out-json", default="storage/artifacts/datasets/state_transition_matrix.json")
    parser.add_argument("--out-md", default="storage/artifacts/datasets/state_transition_matrix.md")
    parser.add_argument("--state-order", default=",".join(DEFAULT_STATE_ORDER))
    args = parser.parse_args()

    state_order = [s.strip().lower() for s in str(args.state_order).split(",") if s.strip()]
    input_path = Path(args.input)
    if not input_path.exists():
        candidates = [
            Path("data/proposal_eval_scenarios_large.jsonl"),
            Path("data/proposal_eval_scenarios.jsonl"),
            Path("storage/artifacts/proposal/latest/scenarios.jsonl"),
        ]
        input_path = next((p for p in candidates if p.exists()), input_path)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {args.input}. "
            "Provide --input <scenario_jsonl> (with turns[] or flat behavior_state rows)."
        )

    rows = load_jsonl(input_path)

    counts_nested = build_counts_nested_turns(rows)
    counts_flat = build_counts_flat(rows)

    # Prefer nested-turns if present; otherwise flat.
    nested_edges = sum(sum(d.values()) for d in counts_nested.values())
    flat_edges = sum(sum(d.values()) for d in counts_flat.values())
    counts = counts_nested if nested_edges >= flat_edges else counts_flat
    edge_count = nested_edges if nested_edges >= flat_edges else flat_edges
    mode = "nested_turns" if nested_edges >= flat_edges else "flat_rows"

    matrix = normalize_counts(counts, state_order)
    matrix_row_major = row_major(matrix, state_order)
    top3 = top_next(matrix, top_k=3)

    out = {
        "mode": mode,
        "input": str(input_path),
        "state_order": state_order,
        "edge_count": edge_count,
        "matrix": matrix,
        "row_major": matrix_row_major,
        "transitions": top3,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    lines: List[str] = []
    lines.append("# State Transition Matrix")
    lines.append("")
    lines.append(f"- Input: `{input_path}`")
    lines.append(f"- Mode: `{mode}`")
    lines.append(f"- Edges: `{edge_count}`")
    lines.append("")
    lines.append("## Top-3 Next States")
    for src in state_order:
        nxt = ", ".join(top3.get(src, [])) or "-"
        lines.append(f"- `{src}` -> {nxt}")
    lines.append("")
    lines.append("## Row-Major Values")
    lines.append("```text")
    lines.append(", ".join(f"{v:.6f}" for v in matrix_row_major))
    lines.append("```")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"saved_json={out_json}")
    print(f"saved_md={out_md}")


if __name__ == "__main__":
    main()

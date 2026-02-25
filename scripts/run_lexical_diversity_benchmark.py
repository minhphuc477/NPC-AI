#!/usr/bin/env python3
"""Compute lexical-diversity benchmark from proposal run responses."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

TOKEN_RE = re.compile(r"[a-z0-9']+")
STOPWORDS = {
    "the",
    "a",
    "an",
    "to",
    "in",
    "of",
    "and",
    "is",
    "are",
    "i",
    "you",
    "we",
    "it",
    "that",
    "this",
    "for",
    "with",
    "on",
    "at",
    "be",
    "as",
    "my",
    "your",
    "me",
    "our",
    "can",
    "will",
    "do",
    "does",
    "was",
    "were",
}


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


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall((text or "").lower())


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return float("nan")
    vals = sorted(float(v) for v in values)
    if len(vals) == 1:
        return vals[0]
    idx = (len(vals) - 1) * p
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return vals[lo]
    frac = idx - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def bootstrap_mean_ci(values: Sequence[float], seed: int, iters: int = 3000) -> Dict[str, float]:
    vals = [float(v) for v in values if not math.isnan(float(v))]
    if not vals:
        return {
            "n": 0,
            "mean": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
        }
    if len(vals) == 1:
        return {"n": 1, "mean": vals[0], "ci95_low": vals[0], "ci95_high": vals[0]}

    rng = random.Random(seed)
    means: List[float] = []
    n = len(vals)
    for _ in range(iters):
        draw = [vals[rng.randrange(n)] for __ in range(n)]
        means.append(statistics.fmean(draw))
    return {
        "n": n,
        "mean": statistics.fmean(vals),
        "ci95_low": percentile(means, 0.025),
        "ci95_high": percentile(means, 0.975),
    }


def paired_bootstrap_delta_ci(
    target_values: Sequence[float],
    baseline_values: Sequence[float],
    seed: int,
    iters: int = 3000,
) -> Dict[str, float]:
    n = min(len(target_values), len(baseline_values))
    if n == 0:
        return {
            "n": 0,
            "mean_delta": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "p_delta_le_0": float("nan"),
        }
    deltas = [float(target_values[i]) - float(baseline_values[i]) for i in range(n)]
    if n == 1:
        d = deltas[0]
        return {"n": 1, "mean_delta": d, "ci95_low": d, "ci95_high": d, "p_delta_le_0": 1.0 if d <= 0.0 else 0.0}

    rng = random.Random(seed)
    sampled: List[float] = []
    for _ in range(iters):
        draw = [deltas[rng.randrange(n)] for __ in range(n)]
        sampled.append(statistics.fmean(draw))
    return {
        "n": n,
        "mean_delta": statistics.fmean(deltas),
        "ci95_low": percentile(sampled, 0.025),
        "ci95_high": percentile(sampled, 0.975),
        "p_delta_le_0": sum(1 for x in sampled if x <= 0.0) / float(len(sampled)),
    }


def mtld(tokens: Sequence[str], threshold: float = 0.72) -> float:
    toks = list(tokens)
    if not toks:
        return 0.0
    factors = 0.0
    types: set[str] = set()
    count = 0
    for tok in toks:
        count += 1
        types.add(tok)
        ttr = len(types) / float(count)
        if ttr <= threshold:
            factors += 1.0
            types.clear()
            count = 0
    if count > 0:
        ttr = len(types) / float(count)
        if threshold < 1.0:
            factors += (1.0 - ttr) / (1.0 - threshold)
    if factors <= 0.0:
        return float(len(toks))
    return float(len(toks)) / factors


def response_lexical_metrics(response: str) -> Dict[str, float]:
    words = tokenize(response)
    n = len(words)
    if n == 0:
        return {
            "distinct1": 0.0,
            "distinct2": 0.0,
            "content_distinct1": 0.0,
            "mtld": 0.0,
            "repetition_penalty": 1.0,
            "lexical_richness": 0.0,
        }

    distinct1 = len(set(words)) / float(n)
    if n >= 2:
        bigrams = [tuple(words[i : i + 2]) for i in range(n - 1)]
        distinct2 = len(set(bigrams)) / float(len(bigrams))
    else:
        distinct2 = 0.0

    content = [w for w in words if w not in STOPWORDS]
    if content:
        content_distinct1 = len(set(content)) / float(len(content))
    else:
        content_distinct1 = distinct1

    m = mtld(words)
    m_norm = max(0.0, min(1.0, m / 180.0))

    repeats = n - len(set(words))
    repetition_penalty = repeats / float(n)

    lexical_richness = max(
        0.0,
        min(
            1.0,
            0.38 * content_distinct1
            + 0.30 * m_norm
            + 0.20 * distinct2
            + 0.12 * (1.0 - repetition_penalty),
        ),
    )

    return {
        "distinct1": distinct1,
        "distinct2": distinct2,
        "content_distinct1": content_distinct1,
        "mtld": m,
        "repetition_penalty": repetition_penalty,
        "lexical_richness": lexical_richness,
    }


def choose_one_response_per_scenario(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    selected: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        sid = str(row.get("scenario_id", "")).strip()
        if not sid:
            continue
        k = (int(row.get("repeat_index", 999999) or 999999), int(row.get("request_index", 999999999) or 999999999))
        prev = selected.get(sid)
        if prev is None:
            selected[sid] = dict(row)
            selected[sid]["_k"] = k
            continue
        if k < prev.get("_k", (999999, 999999999)):
            selected[sid] = dict(row)
            selected[sid]["_k"] = k
    for sid in list(selected.keys()):
        selected[sid].pop("_k", None)
    return selected


def render_markdown(path: Path, payload: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# Lexical Diversity Benchmark")
    lines.append("")
    lines.append(f"- Run dir: `{payload.get('run_dir', '')}`")
    lines.append(f"- Scenario count: `{payload.get('scenario_count', 0)}`")
    lines.append("")

    lines.append("## Arm Summary")
    lines.append("| Arm | Metric | Mean | 95% CI |")
    lines.append("|---|---|---:|---:|")
    for arm, metrics in payload.get("summary", {}).items():
        for metric, vals in metrics.items():
            lines.append(
                f"| {arm} | {metric} | {vals.get('mean', float('nan')):.4f} | "
                f"({vals.get('ci95_low', float('nan')):.4f}, {vals.get('ci95_high', float('nan')):.4f}) |"
            )
    lines.append("")

    lines.append("## Paired Deltas")
    lines.append("| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) |")
    lines.append("|---|---|---:|---:|---:|")
    for comp, metrics in payload.get("paired_deltas", {}).items():
        for metric, vals in metrics.items():
            lines.append(
                f"| {comp} | {metric} | {vals.get('mean_delta', float('nan')):.4f} | "
                f"({vals.get('ci95_low', float('nan')):.4f}, {vals.get('ci95_high', float('nan')):.4f}) | "
                f"{vals.get('p_delta_le_0', float('nan')):.4f} |"
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lexical-diversity benchmark from proposal response artifacts.")
    parser.add_argument("--run-dir", required=True, help="Proposal run dir (artifacts/proposal/<run_id>)")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument(
        "--metrics",
        default="distinct1,distinct2,content_distinct1,mtld,repetition_penalty,lexical_richness",
    )
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    responses_dir = run_dir / "responses"
    run_config = read_json(run_dir / "run_config.json")
    comparison_plan = read_json(run_dir / "comparison_plan.json")

    arms = [str(a.get("arm_id", "")).strip() for a in run_config.get("arms", []) if str(a.get("arm_id", "")).strip()]
    metrics = [m.strip() for m in str(args.metrics).split(",") if m.strip()]

    per_arm_by_sid: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for arm in arms:
        rows = read_jsonl(responses_dir / f"{arm}.jsonl")
        picked = choose_one_response_per_scenario(rows)
        per_arm_by_sid[arm] = picked

    common_sids: List[str] = sorted(
        sid
        for sid in set().union(*[set(v.keys()) for v in per_arm_by_sid.values()])
        if all(sid in per_arm_by_sid.get(arm, {}) for arm in arms)
    )

    scored: Dict[str, Dict[str, Dict[str, float]]] = {}
    summary: Dict[str, Dict[str, Any]] = {}
    for arm_idx, arm in enumerate(arms):
        by_sid = per_arm_by_sid.get(arm, {})
        arm_scores: Dict[str, Dict[str, float]] = {}
        for sid in common_sids:
            response = str(by_sid.get(sid, {}).get("response", ""))
            arm_scores[sid] = response_lexical_metrics(response)
        scored[arm] = arm_scores

        arm_summary: Dict[str, Any] = {}
        for metric_idx, metric in enumerate(metrics):
            values = [float(arm_scores[sid].get(metric, float("nan"))) for sid in common_sids]
            values = [v for v in values if not math.isnan(v)]
            arm_summary[metric] = bootstrap_mean_ci(values, seed=int(args.seed) + 71 * (arm_idx + 1) + metric_idx)
        summary[arm] = arm_summary

    paired_deltas: Dict[str, Dict[str, Any]] = {}
    for comp_idx, comp in enumerate(comparison_plan.get("comparisons", [])):
        cid = str(comp.get("comparison_id", "")).strip()
        target_arm = str(comp.get("target_arm", "")).strip()
        baseline_arm = str(comp.get("baseline_arm", "")).strip()
        if not cid or target_arm not in scored or baseline_arm not in scored:
            continue
        payload: Dict[str, Any] = {}
        for metric_idx, metric in enumerate(metrics):
            target_vals: List[float] = []
            base_vals: List[float] = []
            for sid in common_sids:
                target_vals.append(float(scored[target_arm][sid].get(metric, 0.0)))
                base_vals.append(float(scored[baseline_arm][sid].get(metric, 0.0)))
            payload[metric] = paired_bootstrap_delta_ci(
                target_vals,
                base_vals,
                seed=int(args.seed) + 1701 * (comp_idx + 1) + 41 * (metric_idx + 1),
            )
        paired_deltas[cid] = payload

    out = {
        "run_dir": str(run_dir),
        "scenario_count": len(common_sids),
        "metrics": metrics,
        "summary": summary,
        "paired_deltas": paired_deltas,
    }

    out_json = Path(args.output_json) if str(args.output_json).strip() else run_dir / "lexical_diversity_summary.json"
    out_md = Path(args.output_md) if str(args.output_md).strip() else run_dir / "lexical_diversity_report.md"
    write_json(out_json, out)
    render_markdown(out_md, out)
    print(f"Lexical benchmark written: {out_json}")
    print(f"Lexical report written: {out_md}")


if __name__ == "__main__":
    main()


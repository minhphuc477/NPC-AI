#!/usr/bin/env python3
"""Evaluate retrieval robustness on adversarial protocol ratio splits.

Metrics are pairwise on (query, positive_text, negative_text):
- pair_accuracy: score(query, positive) > score(query, negative)
- adversarial_success_rate: negatives that beat positives for adversarial families
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import sys
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
import numpy as np


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
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_md(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(str(text).lower())


def lexical_score(query: str, doc: str) -> float:
    q = tokenize(query)
    d = tokenize(doc)
    if not q or not d:
        return 0.0
    d_set = set(d)
    overlap = sum(1 for t in q if t in d_set)
    return float(overlap / max(1, len(q)))


def bootstrap_mean(values: Sequence[float], rng: random.Random, n_boot: int = 1000) -> Dict[str, float]:
    cleaned = [float(v) for v in values if isinstance(v, (int, float)) and not math.isnan(float(v))]
    if not cleaned:
        return {"mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    n = len(cleaned)
    means: List[float] = []
    for _ in range(max(100, int(n_boot))):
        sample = [cleaned[rng.randrange(n)] for _ in range(n)]
        means.append(float(sum(sample) / n))
    means.sort()
    lo = means[int(0.025 * (len(means) - 1))]
    hi = means[int(0.975 * (len(means) - 1))]
    return {"mean": float(sum(cleaned) / n), "ci95_low": float(lo), "ci95_high": float(hi)}


def evaluate_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    score_fn,
    name: str,
    seed: int,
) -> Dict[str, Any]:
    per_row: List[Dict[str, Any]] = []
    accuracy_values: List[float] = []
    asr_values: List[float] = []
    by_family: Dict[str, List[Dict[str, Any]]] = {}

    for row in rows:
        query = str(row.get("query", ""))
        pos = str(row.get("positive_text", ""))
        neg = str(row.get("negative_text", ""))
        fam = str(row.get("negative_family", "unknown"))
        s_pos = float(score_fn(query, pos))
        s_neg = float(score_fn(query, neg))
        is_correct = 1.0 if s_pos > s_neg else 0.0
        accuracy_values.append(is_correct)
        is_adv_family = fam in {"trust_spoof", "evidence_poison"}
        asr = 1.0 if (is_adv_family and s_neg >= s_pos) else 0.0
        if is_adv_family:
            asr_values.append(asr)
        rec = {
            "query_id": row.get("query_id"),
            "negative_family": fam,
            "score_positive": s_pos,
            "score_negative": s_neg,
            "correct": bool(is_correct > 0.5),
            "adv_success": bool(asr > 0.5),
        }
        per_row.append(rec)
        by_family.setdefault(fam, []).append(rec)

    rng = random.Random(int(seed))
    fam_payload: Dict[str, Any] = {}
    for fam, fam_rows in sorted(by_family.items()):
        fam_acc = [1.0 if bool(r.get("correct")) else 0.0 for r in fam_rows]
        fam_asr_vals = [1.0 if bool(r.get("adv_success")) else 0.0 for r in fam_rows if fam in {"trust_spoof", "evidence_poison"}]
        fam_payload[fam] = {
            "pair_count": len(fam_rows),
            "pair_accuracy": bootstrap_mean(fam_acc, rng, n_boot=1000),
            "adversarial_success_rate": bootstrap_mean(fam_asr_vals, rng, n_boot=1000) if fam_asr_vals else None,
        }

    return {
        "method": name,
        "pair_count": len(per_row),
        "pair_accuracy": bootstrap_mean(accuracy_values, rng, n_boot=1500),
        "adversarial_success_rate": bootstrap_mean(asr_values, rng, n_boot=1500) if asr_values else None,
        "by_family": fam_payload,
        "per_row": per_row,
    }


def render_md(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Retrieval Protocol Split Evaluation")
    lines.append("")
    lines.append(f"- Protocol summary: `{report['inputs']['protocol_summary']}`")
    lines.append("")
    for split in report.get("splits", []):
        lines.append(f"## Split `{split['split_name']}` ({split['ratio']})")
        lines.append(f"- Pairs: `{split['pair_count']}`")
        lines.append("| Method | Pair Accuracy | Adversarial Success Rate |")
        lines.append("|---|---:|---:|")
        for m in split.get("methods", []):
            acc = m.get("pair_accuracy", {})
            asr = m.get("adversarial_success_rate")
            asr_mean = float("nan")
            if isinstance(asr, dict):
                asr_mean = float(asr.get("mean", float("nan")))
            lines.append(
                f"| {m.get('method')} | {acc.get('mean', float('nan')):.4f} "
                f"[{acc.get('ci95_low', float('nan')):.4f}, {acc.get('ci95_high', float('nan')):.4f}] | "
                f"{asr_mean:.4f} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-summary", default="storage/artifacts/datasets/retrieval_protocol/summary.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--car-checkpoint", default="", help="Optional CAR checkpoint for dense pair scoring.")
    parser.add_argument("--car-device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--car-max-length", type=int, default=256)
    parser.add_argument("--output-json", default="storage/artifacts/benchmarks/retrieval_protocol_split_eval.json")
    parser.add_argument("--output-md", default="storage/artifacts/benchmarks/retrieval_protocol_split_eval.md")
    args = parser.parse_args()

    summary_path = Path(args.protocol_summary)
    summary = read_json(summary_path)

    encoder = None
    if str(args.car_checkpoint).strip():
        from core.car_retriever import CAREncoder, CAREncoderConfig

        encoder = CAREncoder(
            CAREncoderConfig(
                model_name_or_path=str(args.car_checkpoint).strip(),
                device=str(args.car_device),
                max_length=int(args.car_max_length),
            )
        )

    def car_score(query: str, doc: str) -> float:
        assert encoder is not None
        vecs = encoder.encode([query, doc], batch_size=2)
        if vecs.shape[0] < 2:
            return 0.0
        return float(np.dot(vecs[0], vecs[1]))

    splits_out: List[Dict[str, Any]] = []
    for split in summary.get("ratio_splits", []):
        split_path = Path(str(split.get("path", "")))
        if not split_path.exists():
            continue
        rows = read_jsonl(split_path)
        methods: List[Dict[str, Any]] = []
        methods.append(
            evaluate_rows(
                rows,
                score_fn=lexical_score,
                name="bm25_proxy",
                seed=int(args.seed) + 7,
            )
        )
        if encoder is not None:
            methods.append(
                evaluate_rows(
                    rows,
                    score_fn=car_score,
                    name="car_dense",
                    seed=int(args.seed) + 11,
                )
            )
        splits_out.append(
            {
                "split_name": split_path.name,
                "ratio": split.get("ratio", "unknown"),
                "pair_count": len(rows),
                "methods": methods,
            }
        )

    report = {
        "inputs": {
            "protocol_summary": str(summary_path),
            "car_checkpoint": str(args.car_checkpoint),
        },
        "splits": splits_out,
    }
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    write_json(out_json, report)
    write_md(out_md, render_md(report))
    print(f"saved_json={out_json}")
    print(f"saved_md={out_md}")


if __name__ == "__main__":
    main()

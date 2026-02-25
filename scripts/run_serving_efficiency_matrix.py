#!/usr/bin/env python3
"""Benchmark multi-model serving efficiency with quality-normalized analysis."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


import run_publication_benchmark_suite as pubsuite  # type: ignore


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def is_pareto_optimal(points: Dict[str, Tuple[float, float]]) -> Dict[str, bool]:
    # Minimize latency, maximize quality.
    out: Dict[str, bool] = {}
    for a, (lat_a, qual_a) in points.items():
        dominated = False
        if math.isnan(lat_a) or math.isnan(qual_a):
            out[a] = False
            continue
        for b, (lat_b, qual_b) in points.items():
            if a == b:
                continue
            if math.isnan(lat_b) or math.isnan(qual_b):
                continue
            if lat_b <= lat_a and qual_b >= qual_a and (lat_b < lat_a or qual_b > qual_a):
                dominated = True
                break
        out[a] = not dominated
    return out


def best_latency_at_or_above_quality(points: Dict[str, Tuple[float, float]], quality: float) -> float:
    cands = [lat for lat, qual in points.values() if not math.isnan(lat) and not math.isnan(qual) and qual >= quality]
    if not cands:
        return float("nan")
    return min(cands)


def render_report(path: Path, payload: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# Serving Efficiency Matrix Report")
    lines.append("")
    lines.append(f"- Run ID: `{payload.get('run_id', '')}`")
    lines.append(f"- Prompt count: `{payload.get('prompt_count', 0)}`")
    lines.append(f"- Models: `{', '.join(payload.get('models', []))}`")
    lines.append("")

    lines.append("## Summary")
    lines.append("| Model | TTFT ms | Total ms | Tokens/s | BERTScore F1 | QPS-like (quality/s) | Pareto | Frontier ratio |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for model in payload.get("models", []):
        row = payload.get("model_summary", {}).get(model, {})
        lines.append(
            f"| {model} | {row.get('ttft_ms_mean', float('nan')):.3f} | "
            f"{row.get('total_time_ms_mean', float('nan')):.3f} | "
            f"{row.get('tokens_per_s_mean', float('nan')):.3f} | "
            f"{row.get('bertscore_f1_mean', float('nan')):.4f} | "
            f"{row.get('quality_per_second', float('nan')):.4f} | "
            f"{'yes' if row.get('pareto_optimal', False) else 'no'} | "
            f"{row.get('latency_ratio_to_quality_frontier', float('nan')):.4f} |"
        )
    lines.append("")

    candidate = str(payload.get("candidate_model", "")).strip()
    if candidate:
        lines.append("## Candidate Delta vs Baselines")
        lines.append("| Baseline | Delta total ms | Delta BERTScore | Delta quality/s |")
        lines.append("|---|---:|---:|---:|")
        for row in payload.get("candidate_deltas", []):
            lines.append(
                f"| {row.get('baseline_model', '')} | {row.get('delta_total_time_ms', float('nan')):.3f} | "
                f"{row.get('delta_bertscore_f1', float('nan')):.4f} | "
                f"{row.get('delta_quality_per_second', float('nan')):.4f} |"
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-model serving efficiency matrix benchmark.")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--models", default="elara-npc:latest,phi3:mini,phi3:latest")
    parser.add_argument("--candidate-model", default="elara-npc:latest")
    parser.add_argument("--prompts", default="data/serving_prompts_wide.jsonl")
    parser.add_argument("--serving-references", default="data/serving_references_wide.jsonl")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument("--bertscore-lang", default="en")
    parser.add_argument("--output-root", default="artifacts/serving_efficiency")
    args = parser.parse_args()

    models = parse_csv(args.models)
    if len(models) < 2:
        raise ValueError("Need at least 2 models in --models")

    prompts_path = Path(args.prompts)
    refs_path = Path(args.serving_references)
    if not prompts_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompts_path}")

    prompts = pubsuite.read_jsonl(prompts_path)
    references = pubsuite.load_serving_references(refs_path) if refs_path.exists() else {}

    run_id = utc_stamp()
    run_dir = Path(args.output_root) / run_id
    serving_dir = run_dir / "serving"
    serving_dir.mkdir(parents=True, exist_ok=True)

    all_records: Dict[str, List[Dict[str, Any]]] = {m: [] for m in models}

    request_index = 0
    for model in models:
        print(f"[serving-matrix] model={model}")
        for repeat_idx in range(max(1, int(args.repeats))):
            for prompt_row in prompts:
                request_index += 1
                prompt_id = str(prompt_row.get("prompt_id", f"row_{request_index}"))
                prompt = str(prompt_row.get("prompt", ""))
                result = pubsuite.benchmark_ollama_prompt(
                    host=str(args.host),
                    model=model,
                    prompt=prompt,
                    temperature=float(args.temperature),
                    max_tokens=int(args.max_tokens),
                )
                row = {
                    "request_index": request_index,
                    "repeat_index": repeat_idx,
                    "model": model,
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                }
                row.update(result)
                all_records[model].append(row)

    for model, rows in all_records.items():
        write_jsonl(serving_dir / f"requests_{model.replace(':', '_')}.jsonl", rows)

    summary: Dict[str, Dict[str, Any]] = {}
    for model, rows in all_records.items():
        summary[model] = pubsuite.summarize_serving_records(rows, seed=int(args.seed) + len(model) * 11)

    bert_summary = None
    if references:
        bert_summary = pubsuite.summarize_bertscore_records(
            all_serving_records=all_records,
            references=references,
            lang=str(args.bertscore_lang),
            seed=int(args.seed),
        )

    points: Dict[str, Tuple[float, float]] = {}
    for model in models:
        lat = float(summary.get(model, {}).get("total_time_ms", {}).get("mean", float("nan")))
        if bert_summary:
            qual = float(bert_summary.get("models", {}).get(model, {}).get("bertscore_f1", {}).get("mean", float("nan")))
        else:
            # Fallback quality proxy if references are unavailable.
            qual = 1.0 - float(summary.get(model, {}).get("error_rate", float("nan")))
        points[model] = (lat, qual)

    pareto = is_pareto_optimal(points)

    model_summary: Dict[str, Any] = {}
    for model in models:
        lat, qual = points.get(model, (float("nan"), float("nan")))
        ttft = float(summary.get(model, {}).get("ttft_ms", {}).get("mean", float("nan")))
        tps = float(summary.get(model, {}).get("tokens_per_s", {}).get("mean", float("nan")))
        qps = (qual / (lat / 1000.0)) if (not math.isnan(qual) and not math.isnan(lat) and lat > 0.0) else float("nan")
        best_lat_for_qual = best_latency_at_or_above_quality(points, qual) if not math.isnan(qual) else float("nan")
        frontier_ratio = (lat / best_lat_for_qual) if (not math.isnan(lat) and not math.isnan(best_lat_for_qual) and best_lat_for_qual > 0.0) else float("nan")
        model_summary[model] = {
            "ttft_ms_mean": ttft,
            "total_time_ms_mean": lat,
            "tokens_per_s_mean": tps,
            "bertscore_f1_mean": qual,
            "quality_per_second": qps,
            "pareto_optimal": bool(pareto.get(model, False)),
            "latency_ratio_to_quality_frontier": frontier_ratio,
        }

    candidate = str(args.candidate_model)
    candidate_deltas: List[Dict[str, Any]] = []
    if candidate in models:
        cand = model_summary.get(candidate, {})
        for model in models:
            if model == candidate:
                continue
            base = model_summary.get(model, {})
            candidate_deltas.append(
                {
                    "baseline_model": model,
                    "delta_total_time_ms": float(cand.get("total_time_ms_mean", float("nan")))
                    - float(base.get("total_time_ms_mean", float("nan"))),
                    "delta_bertscore_f1": float(cand.get("bertscore_f1_mean", float("nan")))
                    - float(base.get("bertscore_f1_mean", float("nan"))),
                    "delta_quality_per_second": float(cand.get("quality_per_second", float("nan")))
                    - float(base.get("quality_per_second", float("nan"))),
                }
            )

    payload = {
        "run_id": run_id,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "host": args.host,
        "models": models,
        "candidate_model": candidate,
        "prompt_count": len(prompts),
        "inputs": {
            "prompts": str(prompts_path),
            "serving_references": str(refs_path) if refs_path.exists() else "",
        },
        "model_summary": model_summary,
        "candidate_deltas": candidate_deltas,
    }

    if bert_summary is not None:
        payload["bertscore_summary"] = bert_summary

    write_json(run_dir / "summary.json", payload)
    render_report(run_dir / "report.md", payload)
    print(f"Serving efficiency matrix written: {run_dir}")
    print(f"  - {run_dir / 'summary.json'}")
    print(f"  - {run_dir / 'report.md'}")


if __name__ == "__main__":
    main()


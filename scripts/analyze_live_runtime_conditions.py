#!/usr/bin/env python3
"""Aggregate live/interactive runtime metrics from UE5-style benchmark artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    vals = sorted(values)
    if len(vals) == 1:
        return vals[0]
    idx = (len(vals) - 1) * p
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return vals[lo]
    w = idx - lo
    return vals[lo] * (1.0 - w) + vals[hi] * w


def extract_ms(record: Dict[str, Any]) -> float | None:
    if "total_time_ms" in record:
        try:
            return float(record["total_time_ms"])
        except Exception:
            return None
    if "total_duration_ns" in record:
        try:
            return float(record["total_duration_ns"]) / 1e6
        except Exception:
            return None
    return None


def infer_hardware_tier(path: Path) -> str:
    lowered = str(path).lower()
    if "laptop" in lowered:
        return "laptop"
    if "desktop" in lowered or "rtx" in lowered:
        return "desktop_gpu"
    if "proposal" in lowered or "preflight" in lowered:
        return "dev_workstation"
    return "unknown"


def parse_parameter_count(value: Any) -> float:
    try:
        parsed = float(value)
        if parsed > 0:
            return parsed
    except Exception:
        pass
    return float("nan")


def parse_quant_bits(level: str) -> float:
    lowered = str(level or "").lower()
    if "q2" in lowered:
        return 2.0
    if "q3" in lowered:
        return 3.0
    if "q4" in lowered:
        return 4.0
    if "q5" in lowered:
        return 5.0
    if "q6" in lowered:
        return 6.0
    if "q8" in lowered:
        return 8.0
    if "f16" in lowered or "fp16" in lowered:
        return 16.0
    if "f32" in lowered or "fp32" in lowered:
        return 32.0
    return 8.0


def estimate_model_memory_mb(model_entry: Dict[str, Any]) -> float:
    details = model_entry.get("details", {}) if isinstance(model_entry, dict) else {}
    info = model_entry.get("model_info", {}) if isinstance(model_entry, dict) else {}
    parameter_count = parse_parameter_count(info.get("general.parameter_count"))
    if math.isnan(parameter_count):
        return float("nan")
    quant_level = str(details.get("quantization_level", ""))
    bits_per_param = parse_quant_bits(quant_level)
    bytes_est = parameter_count * (bits_per_param / 8.0) * 1.12
    return float(bytes_est / (1024.0 * 1024.0))


def _nanmean(values: List[float]) -> float:
    filtered = [float(v) for v in values if not math.isnan(float(v))]
    return mean(filtered) if filtered else float("nan")


def iter_run_dirs(root: Path) -> Iterable[Path]:
    for op_path in root.rglob("operational_metrics.json"):
        run_dir = op_path.parent
        responses_dir = run_dir / "responses"
        if responses_dir.exists() and responses_dir.is_dir():
            yield run_dir


def summarize_arm(op_arm: Dict[str, Any], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    durations = [x for x in (extract_ms(r) for r in rows) if x is not None]
    ok_rows = [r for r in rows if bool(r.get("ok", False))]
    dropped = [r for r in rows if (not bool(r.get("ok", False))) or bool(r.get("response_fallback", False))]

    return {
        "interaction_count": len(rows),
        "ok_count": len(ok_rows),
        "dropped_interaction_rate": (len(dropped) / len(rows)) if rows else float("nan"),
        "avg_response_time_ms": mean(durations) if durations else float("nan"),
        "p95_response_time_ms": percentile(durations, 0.95),
        "timeout_rate": float(op_arm.get("timeout_rate", float("nan"))),
        "fallback_rate": float(op_arm.get("fallback_rate", float("nan"))),
        "retry_rate": float(op_arm.get("retry_rate", float("nan"))),
        "first_pass_accept_rate": float(op_arm.get("first_pass_accept_rate", float("nan"))),
    }


def extract_run_memory_metadata(run_dir: Path, arm_id: str) -> Dict[str, Any]:
    run_config_path = run_dir / "run_config.json"
    hardware_path = run_dir / "metadata" / "hardware.json"
    models_path = run_dir / "metadata" / "models.json"

    if not run_config_path.exists() or not hardware_path.exists() or not models_path.exists():
        return {
            "model_name": "",
            "estimated_model_memory_mb": float("nan"),
            "host_ram_gb": float("nan"),
        }

    try:
        run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        hardware = json.loads(hardware_path.read_text(encoding="utf-8"))
        models = json.loads(models_path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "model_name": "",
            "estimated_model_memory_mb": float("nan"),
            "host_ram_gb": float("nan"),
        }

    arms = run_config.get("arms", []) if isinstance(run_config, dict) else []
    model_name = ""
    if isinstance(arms, list):
        for arm in arms:
            if not isinstance(arm, dict):
                continue
            if str(arm.get("arm_id", "")) == arm_id:
                model_name = str(arm.get("model", "")).strip()
                break

    model_entry = models.get(model_name, {}) if isinstance(models, dict) else {}
    host_ram_gb = float(hardware.get("ram_gb", float("nan"))) if isinstance(hardware, dict) else float("nan")
    return {
        "model_name": model_name,
        "estimated_model_memory_mb": estimate_model_memory_mb(model_entry),
        "host_ram_gb": host_ram_gb,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze live runtime conditions from existing UE5/proposal artifacts.")
    parser.add_argument("--root", default="releases", help="Root folder to scan for operational_metrics.json")
    parser.add_argument("--arm", default="proposed_contextual_controlled", help="Arm ID to summarize")
    parser.add_argument("--output-json", default="artifacts/publication_profiles/live_runtime_summary.json")
    parser.add_argument("--output-md", default="artifacts/publication_profiles/live_runtime_summary.md")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")

    per_run: List[Dict[str, Any]] = []
    for run_dir in iter_run_dirs(root):
        op_path = run_dir / "operational_metrics.json"
        payload = json.loads(op_path.read_text(encoding="utf-8"))
        op_arm = payload.get("arms", {}).get(args.arm)
        if not isinstance(op_arm, dict):
            continue

        response_file = run_dir / "responses" / f"{args.arm}.jsonl"
        rows = read_jsonl(response_file) if response_file.exists() else []
        summary = summarize_arm(op_arm, rows)
        memory_meta = extract_run_memory_metadata(run_dir, arm_id=args.arm)
        summary.update(
            {
                "run_dir": str(run_dir),
                "hardware_tier": infer_hardware_tier(run_dir),
                **memory_meta,
            }
        )
        per_run.append(summary)

    if not per_run:
        raise RuntimeError(f"No runs found for arm={args.arm} under {root}")

    def _avg(key: str) -> float:
        vals = [float(x.get(key, float("nan"))) for x in per_run if not math.isnan(float(x.get(key, float("nan"))))]
        return mean(vals) if vals else float("nan")

    by_tier: Dict[str, List[Dict[str, Any]]] = {}
    for row in per_run:
        tier = str(row.get("hardware_tier", "unknown"))
        by_tier.setdefault(tier, []).append(row)

    tier_summary: Dict[str, Any] = {}
    for tier, rows in sorted(by_tier.items()):
        tier_summary[tier] = {
            "run_count": len(rows),
            "avg_response_time_ms": _nanmean([float(r.get("avg_response_time_ms", float("nan"))) for r in rows]),
            "p95_response_time_ms": _nanmean([float(r.get("p95_response_time_ms", float("nan"))) for r in rows]),
            "timeout_rate": _nanmean([float(r.get("timeout_rate", float("nan"))) for r in rows]),
            "fallback_rate": _nanmean([float(r.get("fallback_rate", float("nan"))) for r in rows]),
            "retry_rate": _nanmean([float(r.get("retry_rate", float("nan"))) for r in rows]),
            "dropped_interaction_rate": _nanmean([float(r.get("dropped_interaction_rate", float("nan"))) for r in rows]),
            "first_pass_accept_rate": _nanmean([float(r.get("first_pass_accept_rate", float("nan"))) for r in rows]),
            "estimated_model_memory_mb": _nanmean([float(r.get("estimated_model_memory_mb", float("nan"))) for r in rows]),
            "host_ram_gb": _nanmean([float(r.get("host_ram_gb", float("nan"))) for r in rows]),
        }

    result = {
        "arm": args.arm,
        "root": str(root),
        "run_count": len(per_run),
        "overall": {
            "avg_response_time_ms": _avg("avg_response_time_ms"),
            "p95_response_time_ms": _avg("p95_response_time_ms"),
            "timeout_rate": _avg("timeout_rate"),
            "fallback_rate": _avg("fallback_rate"),
            "retry_rate": _avg("retry_rate"),
            "dropped_interaction_rate": _avg("dropped_interaction_rate"),
            "first_pass_accept_rate": _avg("first_pass_accept_rate"),
            "estimated_model_memory_mb": _avg("estimated_model_memory_mb"),
            "host_ram_gb": _avg("host_ram_gb"),
        },
        "by_hardware_tier": tier_summary,
        "per_run": per_run,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    lines: List[str] = []
    lines.append("# Live Runtime Conditions Summary")
    lines.append("")
    lines.append(f"- Arm: `{args.arm}`")
    lines.append(f"- Runs analyzed: `{len(per_run)}`")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    ov = result["overall"]
    lines.append(f"- Avg response time (ms): `{ov['avg_response_time_ms']:.2f}`")
    lines.append(f"- P95 response time (ms): `{ov['p95_response_time_ms']:.2f}`")
    lines.append(f"- Timeout rate: `{ov['timeout_rate']:.4f}`")
    lines.append(f"- Fallback rate: `{ov['fallback_rate']:.4f}`")
    lines.append(f"- Retry rate: `{ov['retry_rate']:.4f}`")
    lines.append(f"- Dropped interaction rate: `{ov['dropped_interaction_rate']:.4f}`")
    lines.append(f"- First-pass accept rate: `{ov['first_pass_accept_rate']:.4f}`")
    lines.append(f"- Estimated model memory footprint (MB): `{ov['estimated_model_memory_mb']:.2f}`")
    lines.append(f"- Host RAM (GB): `{ov['host_ram_gb']:.2f}`")
    lines.append("")
    lines.append("## By Hardware Tier")
    lines.append("")
    lines.append("| Tier | Runs | Avg ms | P95 ms | Timeout | Fallback | Retry | Dropped | First-pass | Model MB | RAM GB |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for tier, vals in tier_summary.items():
        lines.append(
            f"| {tier} | {vals['run_count']} | {vals['avg_response_time_ms']:.2f} | {vals['p95_response_time_ms']:.2f} | "
            f"{vals['timeout_rate']:.4f} | {vals['fallback_rate']:.4f} | {vals['retry_rate']:.4f} | "
            f"{vals['dropped_interaction_rate']:.4f} | {vals['first_pass_accept_rate']:.4f} | "
            f"{vals['estimated_model_memory_mb']:.2f} | {vals['host_ram_gb']:.2f} |"
        )

    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote: {output_json}")
    print(f"Wrote: {output_md}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Check local Ollama model pack suitability for laptop-safe benchmark baselines."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import requests

try:
    from local_model_profiles import (
        BASELINE_PROFILES,
        baseline_profile_choices,
        fetch_ollama_models,
        format_model_csv,
        merge_unique_models,
        parse_model_csv,
        split_available_missing,
    )
except ModuleNotFoundError:
    from scripts.local_model_profiles import (
        BASELINE_PROFILES,
        baseline_profile_choices,
        fetch_ollama_models,
        format_model_csv,
        merge_unique_models,
        parse_model_csv,
        split_available_missing,
    )


SMOKE_PROMPT = (
    "You are a strict city guard NPC. "
    "Player asks for archive access after a theft alarm. "
    "Reply in 2 short in-character sentences."
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def smoke_generate(host: str, model: str, timeout_s: int, max_tokens: int) -> Dict[str, Any]:
    url = f"{host}/api/generate"
    payload = {
        "model": model,
        "prompt": SMOKE_PROMPT,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": max_tokens,
        },
    }
    t0 = time.perf_counter()
    try:
        resp = requests.post(url, json=payload, timeout=timeout_s)
        resp.raise_for_status()
        body = resp.json()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        text = str(body.get("response", "")).strip()
        return {
            "ok": bool(text),
            "latency_ms": elapsed_ms,
            "response_chars": len(text),
            "response_preview": text[:180],
            "error": "",
        }
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "ok": False,
            "latency_ms": elapsed_ms,
            "response_chars": 0,
            "response_preview": "",
            "error": str(exc),
        }


def render_md(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Local LLM Pack Check")
    lines.append("")
    lines.append(f"- Host: `{report['host']}`")
    lines.append(f"- Profile: `{report['profile']}`")
    lines.append(f"- Installed model count: `{report['installed_model_count']}`")
    lines.append(f"- Requested models: `{', '.join(report['requested_models'])}`")
    lines.append(f"- Missing requested models: `{', '.join(report['missing_models']) if report['missing_models'] else 'none'}`")
    lines.append("")
    lines.append("## Smoke Results")
    for row in report.get("smoke", []):
        lines.append(
            f"- `{row['model']}`: ok={row['ok']} latency_ms={row['latency_ms']:.1f} error={row['error'] or 'none'}"
        )
    lines.append("")
    lines.append("## Recommended")
    lines.append(f"- Baseline CSV: `{report['recommended_baseline_models_csv']}`")
    lines.append(f"- Serving CSV: `{report['recommended_serving_models_csv']}`")
    lines.append("")
    lines.append("## Example")
    lines.append(
        f"`python scripts/run_laptop_full_results.py --baseline-models \"{report['recommended_baseline_models_csv']}\"`"
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--profile", default="laptop_safe", choices=baseline_profile_choices())
    parser.add_argument("--candidate-model", default="elara-npc:latest")
    parser.add_argument("--extra-models", default="")
    parser.add_argument("--timeout-s", type=int, default=20)
    parser.add_argument("--smoke-max-tokens", type=int, default=48)
    parser.add_argument("--max-acceptable-latency-ms", type=float, default=20000.0)
    parser.add_argument("--fail-if-offline", action="store_true")
    parser.add_argument("--output-json", default="storage/artifacts/publication_profiles/local_llm_pack_check.json")
    parser.add_argument("--output-md", default="storage/artifacts/publication_profiles/local_llm_pack_check.md")
    args = parser.parse_args()

    requested = merge_unique_models(
        BASELINE_PROFILES.get(str(args.profile).strip().lower(), []),
        parse_model_csv(str(args.extra_models)),
    )
    if not requested:
        requested = ["phi3:mini"]

    try:
        installed = fetch_ollama_models(str(args.host), timeout_s=max(3, int(args.timeout_s)))
    except Exception as exc:
        if bool(args.fail_if_offline):
            raise
        report = {
            "host": str(args.host),
            "profile": str(args.profile),
            "ollama_available": False,
            "error": str(exc),
            "installed_model_count": 0,
            "requested_models": requested,
            "available_models": [],
            "missing_models": requested,
            "smoke": [],
            "recommended_baseline_models_csv": format_model_csv(requested),
            "recommended_serving_models_csv": format_model_csv([str(args.candidate_model)] + requested),
        }
        out_json = Path(args.output_json)
        out_md = Path(args.output_md)
        write_json(out_json, report)
        write_text(out_md, render_md(report))
        print(f"[warn] Ollama unavailable at {args.host}: {exc}")
        print(f"Saved: {out_json}")
        print(f"Saved: {out_md}")
        print(f"recommended_baseline_models={report['recommended_baseline_models_csv']}")
        return

    available, missing = split_available_missing(requested, installed)

    smoke_rows: List[Dict[str, Any]] = []
    for model in available:
        result = smoke_generate(
            host=str(args.host),
            model=model,
            timeout_s=max(3, int(args.timeout_s)),
            max_tokens=max(8, int(args.smoke_max_tokens)),
        )
        smoke_rows.append({"model": model, **result})

    recommended_baselines = [
        row["model"]
        for row in smoke_rows
        if bool(row.get("ok")) and float(row.get("latency_ms", 9e9)) <= float(args.max_acceptable_latency_ms)
    ]
    if not recommended_baselines:
        recommended_baselines = [row["model"] for row in smoke_rows if bool(row.get("ok"))]
    if not recommended_baselines:
        recommended_baselines = available[:]

    serving_models = merge_unique_models([str(args.candidate_model)], recommended_baselines)

    report = {
        "host": str(args.host),
        "profile": str(args.profile),
        "ollama_available": True,
        "installed_model_count": len(installed),
        "requested_models": requested,
        "available_models": available,
        "missing_models": missing,
        "smoke": smoke_rows,
        "recommended_baseline_models_csv": format_model_csv(recommended_baselines),
        "recommended_serving_models_csv": format_model_csv(serving_models),
    }

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    write_json(out_json, report)
    write_text(out_md, render_md(report))
    print(f"Saved: {out_json}")
    print(f"Saved: {out_md}")
    print(f"recommended_baseline_models={report['recommended_baseline_models_csv']}")


if __name__ == "__main__":
    main()

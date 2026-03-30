#!/usr/bin/env python3
"""Run a project-level scientific readiness audit.

This script validates whether key experimental claims are backed by artifacts:
1) Retrieval ablation size (>= min labeled queries)
2) Operational metrics schema completeness (p90/p95 + retry-count distribution)
3) Novelty benchmark deltas for alpha(s), alpha(s,q), and tau(s)

Outputs:
  - JSON report
  - Markdown summary
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_finite(value: Any) -> bool:
    return _is_number(value) and math.isfinite(float(value))


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    rank = max(0.0, min(1.0, p)) * (len(xs) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return xs[lo]
    frac = rank - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def latest_run_with_operational_metrics(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    runs = sorted(
        p for p in root.iterdir()
        if p.is_dir() and (p / "operational_metrics.json").exists()
    )
    return runs[-1] if runs else None


def _extract_latency_ms(row: Dict[str, Any]) -> Optional[float]:
    ns = row.get("total_duration_ns")
    if isinstance(ns, (int, float)) and not isinstance(ns, bool) and math.isfinite(float(ns)) and float(ns) > 0.0:
        return float(ns) / 1_000_000.0
    ms = row.get("total_duration_ms")
    if isinstance(ms, (int, float)) and not isinstance(ms, bool) and math.isfinite(float(ms)) and float(ms) > 0.0:
        return float(ms)
    return None


def _extract_retry_count(row: Dict[str, Any]) -> int:
    attempts = row.get("rewrite_attempts")
    if isinstance(attempts, int) and attempts >= 0:
        return attempts
    successful = row.get("rewrite_successful_attempts")
    if isinstance(successful, int) and successful >= 0:
        return successful
    attempted = row.get("rewrite_attempted")
    if isinstance(attempted, bool):
        return 1 if attempted else 0
    return 0


def _find_response_file_for_arm(run_dir: Path, arm_id: str) -> Optional[Path]:
    responses_dir = run_dir / "responses"
    if not responses_dir.exists():
        return None
    direct = responses_dir / f"{arm_id}.jsonl"
    if direct.exists():
        return direct
    files = sorted(p for p in responses_dir.glob("*.jsonl") if p.is_file())
    return files[0] if files else None


def backfill_operational_for_arm(
    operational_payload: Dict[str, Any],
    run_dir: Path,
    arm_id: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    payload = json.loads(json.dumps(operational_payload))
    arms = payload.setdefault("arms", {})
    if not isinstance(arms, dict):
        arms = {}
        payload["arms"] = arms

    selected_arm_id = arm_id if arm_id in arms else (next(iter(arms.keys())) if arms else arm_id)
    arm_obj = arms.get(selected_arm_id)
    if not isinstance(arm_obj, dict):
        arm_obj = {}
        arms[selected_arm_id] = arm_obj

    latency = arm_obj.get("latency_ms")
    has_p90 = isinstance(latency, dict) and _is_finite(latency.get("p90"))
    has_p95 = isinstance(latency, dict) and _is_finite(latency.get("p95"))
    retry_dist = arm_obj.get("retry_count_distribution")
    has_retry = isinstance(retry_dist, dict) and isinstance(retry_dist.get("rows"), list) and bool(retry_dist.get("rows"))
    already_complete = has_p90 and has_p95 and has_retry
    details: Dict[str, Any] = {
        "selected_arm_id": selected_arm_id,
        "source": "existing",
        "applied": False,
        "notes": [],
    }
    if already_complete:
        return payload, details

    response_file = _find_response_file_for_arm(run_dir, selected_arm_id)
    if response_file is None:
        details["notes"].append("responses directory or response file not found for backfill")
        return payload, details

    rows = read_jsonl(response_file)
    if not rows:
        details["notes"].append("response file is empty; cannot backfill operational metrics")
        return payload, details

    latencies_ms: List[float] = []
    retries: List[int] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        latency_ms = _extract_latency_ms(row)
        if latency_ms is not None and math.isfinite(latency_ms) and latency_ms > 0.0:
            latencies_ms.append(latency_ms)
        retries.append(max(0, _extract_retry_count(row)))

    if latencies_ms:
        arm_obj["latency_ms"] = {
            "count": len(latencies_ms),
            "mean": float(sum(latencies_ms) / len(latencies_ms)),
            "min": float(min(latencies_ms)),
            "p50": float(percentile(latencies_ms, 0.50)),
            "p90": float(percentile(latencies_ms, 0.90)),
            "p95": float(percentile(latencies_ms, 0.95)),
            "max": float(max(latencies_ms)),
            "source": "backfilled_from_responses",
        }
    else:
        details["notes"].append("no finite latency found in response rows")

    if retries:
        total = len(retries)
        counts = Counter(retries)
        arm_obj["retry_count_distribution"] = {
            "rows": [
                {
                    "retry_count": int(k),
                    "count": int(v),
                    "fraction": float(v / total),
                }
                for k, v in sorted(counts.items())
            ],
            "source": "backfilled_from_responses",
        }

    details["source"] = str(response_file)
    details["applied"] = True
    return payload, details


def summarize_retrieval_gold(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    query_ids = {
        str(row.get("query_id", "")).strip()
        for row in rows
        if str(row.get("query_id", "")).strip()
    }
    query_types: Dict[str, int] = {}
    for row in rows:
        qtype = str(row.get("query_type", "unknown")).strip() or "unknown"
        query_types[qtype] = query_types.get(qtype, 0) + 1

    return {
        "row_count": len(rows),
        "unique_query_count": len(query_ids),
        "query_type_counts": dict(sorted(query_types.items())),
    }


def check_operational_metrics_schema(
    payload: Dict[str, Any],
    arm_id: str,
) -> Dict[str, Any]:
    arms = payload.get("arms", {}) if isinstance(payload, dict) else {}
    if not isinstance(arms, dict) or not arms:
        return {
            "arm_id": arm_id,
            "available_arms": [],
            "selected_arm_id": "",
            "has_latency_summary": False,
            "has_p90": False,
            "has_p95": False,
            "has_retry_distribution": False,
            "retry_rows_count": 0,
            "notes": ["operational_metrics has no populated arms object"],
        }

    selected_arm_id = arm_id if arm_id in arms else next(iter(arms.keys()))
    arm = arms.get(selected_arm_id, {}) if isinstance(arms.get(selected_arm_id), dict) else {}
    latency = arm.get("latency_ms", {}) if isinstance(arm, dict) else {}
    retry_dist = arm.get("retry_count_distribution", {}) if isinstance(arm, dict) else {}
    retry_rows = retry_dist.get("rows", []) if isinstance(retry_dist, dict) else []

    has_latency_summary = isinstance(latency, dict) and bool(latency)
    has_p90 = _is_finite(latency.get("p90")) if isinstance(latency, dict) else False
    has_p95 = _is_finite(latency.get("p95")) if isinstance(latency, dict) else False
    has_retry_distribution = isinstance(retry_rows, list) and len(retry_rows) > 0

    notes: List[str] = []
    if not has_latency_summary:
        notes.append("latency_ms summary is missing for selected arm")
    else:
        if not has_p90:
            notes.append("latency_ms.p90 missing or non-finite")
        if not has_p95:
            notes.append("latency_ms.p95 missing or non-finite")
    if not has_retry_distribution:
        notes.append("retry_count_distribution.rows missing or empty")

    return {
        "arm_id": arm_id,
        "available_arms": sorted(arms.keys()),
        "selected_arm_id": selected_arm_id,
        "has_latency_summary": has_latency_summary,
        "has_p90": has_p90,
        "has_p95": has_p95,
        "has_retry_distribution": has_retry_distribution,
        "retry_rows_count": len(retry_rows) if isinstance(retry_rows, list) else 0,
        "notes": notes,
    }


def check_novelty_benchmark(payload: Dict[str, Any]) -> Dict[str, Any]:
    retrieval_state_rows = payload.get("retrieval_alpha_state_conditioned", [])
    response_rows = payload.get("response_tau_state_conditioned", [])
    query_aware = payload.get("retrieval_query_aware_fusion", {})

    state_deltas = [
        float(row.get("delta_top1_accuracy", 0.0))
        for row in retrieval_state_rows
        if isinstance(row, dict) and _is_number(row.get("delta_top1_accuracy", 0.0))
    ]
    has_state_gain = any(delta > 0.0 for delta in state_deltas)

    query_aware_delta = float(query_aware.get("delta_top1_accuracy", 0.0)) if isinstance(query_aware, dict) else 0.0
    has_query_aware_gain = query_aware_delta > 0.0

    per_state: Dict[str, Dict[str, Optional[float]]] = {}
    for row in response_rows:
        if not isinstance(row, dict):
            continue
        state = str(row.get("state", "unknown"))
        risk = str(row.get("risk", "unknown")).lower()
        delta = row.get("delta_first_pass_accept")
        if not _is_number(delta):
            continue
        if state not in per_state:
            per_state[state] = {"low": None, "high": None}
        if risk in ("low", "high") and isinstance(delta, (int, float)) and not isinstance(delta, bool):
            per_state[state][risk] = float(delta)

    tau_pattern_pass = True
    for state, row in per_state.items():
        low = row.get("low")
        high = row.get("high")
        if low is None or high is None:
            tau_pattern_pass = False
            continue
        if not (low > 0.0 and high < 0.0):
            tau_pattern_pass = False

    notes: List[str] = []
    if not has_state_gain:
        notes.append("No positive delta found in retrieval_alpha_state_conditioned")
    if not has_query_aware_gain:
        notes.append("No positive delta found in retrieval_query_aware_fusion")
    if not tau_pattern_pass:
        notes.append("tau(s) risk polarity check failed (need low-risk positive and high-risk negative deltas)")

    return {
        "state_conditioned_alpha_gain": has_state_gain,
        "query_aware_alpha_gain": has_query_aware_gain,
        "query_aware_delta_top1": query_aware_delta,
        "tau_state_risk_polarity_pass": tau_pattern_pass,
        "tau_state_rows": per_state,
        "notes": notes,
    }


def check_adversarial_protocol_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
    total_pairs = int(payload.get("total_pairs", 0)) if isinstance(payload, dict) else 0
    family_counts = payload.get("family_counts", {}) if isinstance(payload, dict) else {}
    if not isinstance(family_counts, dict):
        family_counts = {}
    bm25_count = int(family_counts.get("bm25_hard_negative", 0))
    trust_count = int(family_counts.get("trust_spoof", 0))
    poison_count = int(family_counts.get("evidence_poison", 0))
    has_all_families = bm25_count > 0 and trust_count > 0 and poison_count > 0
    notes: List[str] = []
    if total_pairs <= 0:
        notes.append("Adversarial protocol summary has zero pairs.")
    if not has_all_families:
        notes.append("Adversarial protocol is missing one or more required families (bm25/trust_spoof/evidence_poison).")
    return {
        "total_pairs": total_pairs,
        "family_counts": family_counts,
        "has_required_families": has_all_families,
        "notes": notes,
    }


def build_recommendations(
    retrieval_ok: bool,
    operational_ok: bool,
    novelty_ok: bool,
    protocol_ok: bool,
    min_queries: int,
) -> List[str]:
    recs: List[str] = []
    if not retrieval_ok:
        recs.append(
            f"Expand retrieval labels to >= {min_queries} unique queries and re-run retrieval evaluation."
        )
        recs.append(
            "Suggested: python scripts/pipeline1_retrieval_labels.py --seed_queries <...> --corpus <...>"
        )
    if not operational_ok:
        recs.append(
            "Re-run proposal alignment evaluation to regenerate operational metrics with latency p90/p95 and retry distribution."
        )
        recs.append(
            "Suggested: python scripts/run_proposal_alignment_eval.py --output-root storage/artifacts/proposal"
        )
    if not novelty_ok:
        recs.append(
            "Re-run novelty benchmark and inspect alpha(s), alpha(s,q), tau(s) deltas."
        )
        recs.append(
            "Suggested: cpp/build/Release/bench_state_conditioned_novelty.exe --output storage/artifacts/benchmarks/state_conditioned_novelty.json"
        )
    if not protocol_ok:
        recs.append(
            "Build the NPC-adversarial corpus protocol package and verify family coverage."
        )
        recs.append(
            "Suggested: python scripts/run_npc_adversarial_protocol.py --retrieval-gold data/retrieval_gold.jsonl --retrieval-corpus data/retrieval_corpus.jsonl"
        )
    if not recs:
        recs.append("Scientific readiness checks passed. Proceed to full benchmark + significance rerun.")
    return recs


def render_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Scientific Readiness Audit")
    lines.append("")
    lines.append(f"- Generated (UTC): `{report['generated_utc']}`")
    lines.append(f"- Overall pass: `{report['overall_pass']}`")
    lines.append("")

    retrieval = report["checks"]["retrieval"]
    lines.append("## Retrieval")
    lines.append(f"- Min required queries: `{retrieval['min_required_queries']}`")
    lines.append(f"- Unique labeled queries: `{retrieval['unique_query_count']}`")
    lines.append(f"- Pass: `{retrieval['pass']}`")
    lines.append("- Query type counts:")
    for qtype, count in retrieval.get("query_type_counts", {}).items():
        lines.append(f"  - `{qtype}`: {count}")
    lines.append("")

    op = report["checks"]["operational"]
    lines.append("## Operational")
    lines.append(f"- Selected arm: `{op.get('selected_arm_id', '')}`")
    lines.append(f"- Has latency summary: `{op['has_latency_summary']}`")
    lines.append(f"- Has p90: `{op['has_p90']}`")
    lines.append(f"- Has p95: `{op['has_p95']}`")
    lines.append(f"- Has retry distribution: `{op['has_retry_distribution']}`")
    lines.append(f"- Retry rows count: `{op['retry_rows_count']}`")
    backfill = op.get("backfill", {}) if isinstance(op.get("backfill"), dict) else {}
    if backfill:
        lines.append(f"- Backfill applied: `{bool(backfill.get('applied'))}`")
        if backfill.get("source"):
            lines.append(f"- Backfill source: `{backfill.get('source')}`")
        if backfill.get("persisted_path"):
            lines.append(f"- Backfill persisted: `{backfill.get('persisted_path')}`")
        notes = backfill.get("notes", [])
        if isinstance(notes, list) and notes:
            lines.append("- Backfill notes:")
            for note in notes:
                lines.append(f"  - {note}")
    if op.get("notes"):
        lines.append("- Notes:")
        for note in op["notes"]:
            lines.append(f"  - {note}")
    lines.append("")

    novelty = report["checks"]["novelty"]
    lines.append("## Novelty")
    lines.append(f"- alpha(s) gain present: `{novelty['state_conditioned_alpha_gain']}`")
    lines.append(f"- alpha(s,q) gain present: `{novelty['query_aware_alpha_gain']}`")
    lines.append(f"- alpha(s,q) delta_top1: `{novelty['query_aware_delta_top1']}`")
    lines.append(f"- tau(s) risk polarity pass: `{novelty['tau_state_risk_polarity_pass']}`")
    if novelty.get("notes"):
        lines.append("- Notes:")
        for note in novelty["notes"]:
            lines.append(f"  - {note}")
    lines.append("")

    protocol = report["checks"]["adversarial_protocol"]
    lines.append("## Adversarial Protocol")
    lines.append(f"- Total pairs: `{protocol['total_pairs']}`")
    lines.append(f"- Has required families: `{protocol['has_required_families']}`")
    if protocol.get("notes"):
        lines.append("- Notes:")
        for note in protocol["notes"]:
            lines.append(f"  - {note}")
    lines.append("")

    lines.append("## Recommendations")
    for rec in report.get("recommendations", []):
        lines.append(f"- {rec}")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-gold", default="data/retrieval_gold.jsonl")
    parser.add_argument("--min-queries", type=int, default=50)
    parser.add_argument("--proposal-root", default="storage/artifacts/proposal")
    parser.add_argument("--operational-run", default="")
    parser.add_argument("--operational-arm", default="proposed_contextual_controlled")
    parser.add_argument(
        "--persist-operational-backfill",
        action="store_true",
        help="Write run-local operational_metrics.backfilled.json when audit backfill is applied.",
    )
    parser.add_argument("--novelty-benchmark", default="storage/artifacts/benchmarks/state_conditioned_novelty.json")
    parser.add_argument("--adversarial-protocol-summary", default="storage/artifacts/datasets/retrieval_protocol/summary.json")
    parser.add_argument(
        "--output-json",
        default="storage/artifacts/publication_profiles/scientific_readiness_audit.json",
    )
    parser.add_argument(
        "--output-md",
        default="storage/artifacts/publication_profiles/scientific_readiness_audit.md",
    )
    args = parser.parse_args()

    retrieval_gold_path = Path(args.retrieval_gold)
    novelty_path = Path(args.novelty_benchmark)
    protocol_path = Path(args.adversarial_protocol_summary)
    proposal_root = Path(args.proposal_root)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    min_queries = max(1, int(args.min_queries))

    if not retrieval_gold_path.exists():
        raise FileNotFoundError(f"retrieval gold file not found: {retrieval_gold_path}")
    if not novelty_path.exists():
        raise FileNotFoundError(f"novelty benchmark file not found: {novelty_path}")
    if not protocol_path.exists():
        raise FileNotFoundError(f"adversarial protocol summary not found: {protocol_path}")

    retrieval_rows = read_jsonl(retrieval_gold_path)
    retrieval_summary = summarize_retrieval_gold(retrieval_rows)
    retrieval_pass = int(retrieval_summary["unique_query_count"]) >= min_queries

    operational_run_path: Optional[Path]
    if str(args.operational_run).strip():
        operational_run_path = Path(args.operational_run)
    else:
        operational_run_path = latest_run_with_operational_metrics(proposal_root)

    if operational_run_path is None:
        op_check = {
            "arm_id": args.operational_arm,
            "available_arms": [],
            "selected_arm_id": "",
            "has_latency_summary": False,
            "has_p90": False,
            "has_p95": False,
            "has_retry_distribution": False,
            "retry_rows_count": 0,
            "notes": ["No operational run with operational_metrics.json found"],
        }
    else:
        operational_payload = read_json(operational_run_path / "operational_metrics.json")
        operational_payload, backfill_info = backfill_operational_for_arm(
            operational_payload,
            operational_run_path,
            args.operational_arm,
        )
        op_check = check_operational_metrics_schema(operational_payload, args.operational_arm)
        op_check["backfill"] = backfill_info
        op_check["run_dir"] = str(operational_run_path)
        if backfill_info.get("applied") and args.persist_operational_backfill:
            backfill_path = operational_run_path / "operational_metrics.backfilled.json"
            write_json(backfill_path, operational_payload)
            op_check["backfill"]["persisted_path"] = str(backfill_path)

    operational_pass = bool(op_check.get("has_p90")) and bool(op_check.get("has_p95")) and bool(
        op_check.get("has_retry_distribution")
    )

    novelty_payload = read_json(novelty_path)
    novelty_check = check_novelty_benchmark(novelty_payload)
    novelty_pass = bool(novelty_check["state_conditioned_alpha_gain"]) and bool(
        novelty_check["query_aware_alpha_gain"]
    ) and bool(novelty_check["tau_state_risk_polarity_pass"])

    protocol_payload = read_json(protocol_path)
    protocol_check = check_adversarial_protocol_summary(protocol_payload)
    protocol_pass = bool(protocol_check["total_pairs"] > 0) and bool(protocol_check["has_required_families"])

    overall_pass = retrieval_pass and operational_pass and novelty_pass and protocol_pass
    report: Dict[str, Any] = {
        "generated_utc": utc_iso(),
        "overall_pass": overall_pass,
        "paths": {
            "retrieval_gold": str(retrieval_gold_path),
            "novelty_benchmark": str(novelty_path),
            "adversarial_protocol_summary": str(protocol_path),
            "proposal_root": str(proposal_root),
            "operational_run": str(operational_run_path) if operational_run_path else "",
        },
        "checks": {
            "retrieval": {
                **retrieval_summary,
                "min_required_queries": min_queries,
                "pass": retrieval_pass,
            },
            "operational": {
                **op_check,
                "pass": operational_pass,
            },
            "novelty": {
                **novelty_check,
                "pass": novelty_pass,
            },
            "adversarial_protocol": {
                **protocol_check,
                "pass": protocol_pass,
            },
        },
        "recommendations": build_recommendations(
            retrieval_ok=retrieval_pass,
            operational_ok=operational_pass,
            novelty_ok=novelty_pass,
            protocol_ok=protocol_pass,
            min_queries=min_queries,
        ),
    }

    write_json(output_json, report)
    write_text(output_md, render_markdown(report))

    print(f"Scientific readiness audit saved: {output_json}")
    print(f"Scientific readiness markdown: {output_md}")
    print(f"overall_pass={overall_pass}")


if __name__ == "__main__":
    main()

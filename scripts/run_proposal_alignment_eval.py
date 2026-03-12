#!/usr/bin/env python3
"""Run proposal-aligned NPC dialogue evaluation and publish reproducible artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, fields, replace
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.response_controller import (
    ControlConfig,
    control_response,
    grounded_fallback_response,
    sanitize_response,
)

TOKEN_RE = re.compile(r"[a-z0-9']+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
MODEL_ID_RE = re.compile(r"[^a-z0-9]+")

BASE_METRICS: Tuple[str, ...] = (
    "context_relevance",
    "persona_consistency",
    "naturalness",
)
GAMEPLAY_METRICS: Tuple[str, ...] = (
    "quest_state_correctness",
    "lore_consistency",
    "multi_turn_contradiction_safety",
    "objective_completion_support",
    "gameplay_usefulness",
    "time_pressure_acceptability",
)
DIAGNOSTIC_METRICS: Tuple[str, ...] = (
    "context_keyword_coverage",
    "context_overlap",
    "persona_keyword_coverage",
    "persona_style",
    "distinct1",
    "length_score",
    "sentence_score",
)


@dataclass(frozen=True)
class EvaluationArm:
    arm_id: str
    model: str
    include_dynamic_context: bool
    use_response_control: bool = False
    control_profile: str = "none"


def parse_list_arg(raw: str) -> List[str]:
    items = [x.strip() for x in re.split(r"[,\n;]+", raw) if x.strip()]
    out: List[str] = []
    seen = set()
    for item in items:
        lowered = item.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        out.append(item)
    return out


def sanitize_model_id(model: str) -> str:
    lowered = model.lower().strip()
    lowered = MODEL_ID_RE.sub("_", lowered).strip("_")
    if not lowered:
        return "model"
    return lowered


def normalize_control_config(config: ControlConfig) -> ControlConfig:
    return replace(
        config,
        min_context_coverage=max(0.0, min(1.0, float(config.min_context_coverage))),
        min_persona_coverage=max(0.0, min(1.0, float(config.min_persona_coverage))),
        min_response_tokens=max(1, int(config.min_response_tokens)),
        rewrite_temperature=max(0.0, float(config.rewrite_temperature)),
        rewrite_max_tokens=max(8, int(config.rewrite_max_tokens)),
        rewrite_candidates=max(1, int(config.rewrite_candidates)),
        rewrite_temperature_step=max(0.0, float(config.rewrite_temperature_step)),
        early_stop_score=max(0.0, min(1.0, float(config.early_stop_score))),
        relaxed_context_coverage=max(0.0, min(1.0, float(config.relaxed_context_coverage))),
        relaxed_persona_coverage=max(0.0, min(1.0, float(config.relaxed_persona_coverage))),
        relaxed_candidate_score=max(0.0, min(1.0, float(config.relaxed_candidate_score))),
        min_rewrite_gain=max(0.0, float(config.min_rewrite_gain)),
        adaptive_candidate_score=max(0.0, min(1.0, float(config.adaptive_candidate_score))),
        adaptive_context_coverage=max(0.0, min(1.0, float(config.adaptive_context_coverage))),
        adaptive_persona_coverage=max(0.0, min(1.0, float(config.adaptive_persona_coverage))),
        adaptive_high_confidence_score=max(0.0, min(1.0, float(config.adaptive_high_confidence_score))),
        adaptive_mid_confidence_score=max(0.0, min(1.0, float(config.adaptive_mid_confidence_score))),
        adaptive_high_confidence_rewrites=max(1, int(config.adaptive_high_confidence_rewrites)),
        adaptive_mid_confidence_rewrites=max(1, int(config.adaptive_mid_confidence_rewrites)),
        adaptive_low_confidence_rewrites=max(1, int(config.adaptive_low_confidence_rewrites)),
        low_confidence_retry_min_score_gain=max(0.0, float(config.low_confidence_retry_min_score_gain)),
        low_confidence_retry_min_coverage_gain=max(0.0, float(config.low_confidence_retry_min_coverage_gain)),
        latency_relax_start_pressure=max(0.0, min(1.0, float(config.latency_relax_start_pressure))),
        latency_relax_max_delta=max(0.0, min(0.25, float(config.latency_relax_max_delta))),
        low_risk_context_relax=max(0.0, min(0.25, float(config.low_risk_context_relax))),
        low_risk_persona_relax=max(0.0, min(0.25, float(config.low_risk_persona_relax))),
        low_risk_candidate_score_relax=max(0.0, min(0.25, float(config.low_risk_candidate_score_relax))),
        high_risk_context_tighten=max(0.0, min(0.25, float(config.high_risk_context_tighten))),
        high_risk_persona_tighten=max(0.0, min(0.25, float(config.high_risk_persona_tighten))),
        high_risk_candidate_score_tighten=max(0.0, min(0.25, float(config.high_risk_candidate_score_tighten))),
        intent_focus_min_keep=max(1, int(config.intent_focus_min_keep)),
        intent_focus_keep_ratio_low=max(0.2, min(1.0, float(config.intent_focus_keep_ratio_low))),
        intent_focus_keep_ratio_medium=max(0.2, min(1.0, float(config.intent_focus_keep_ratio_medium))),
        intent_focus_keep_ratio_high=max(0.2, min(1.0, float(config.intent_focus_keep_ratio_high))),
        intent_focus_min_relevance=max(0.0, min(1.0, float(config.intent_focus_min_relevance))),
        near_pass_max_context_gap=max(0.0, min(0.5, float(config.near_pass_max_context_gap))),
        near_pass_max_persona_gap=max(0.0, min(0.5, float(config.near_pass_max_persona_gap))),
        near_pass_score_floor=max(0.0, min(1.0, float(config.near_pass_score_floor))),
    )


def apply_control_overrides(base: ControlConfig, overrides: Dict[str, Any]) -> ControlConfig:
    if not overrides:
        return base
    valid_fields = {f.name: f for f in fields(ControlConfig)}
    updates: Dict[str, Any] = {}
    for key, raw_val in overrides.items():
        if key not in valid_fields:
            raise ValueError(f"Unsupported control override key: {key}")
        current_val = getattr(base, key)
        if isinstance(current_val, bool):
            if isinstance(raw_val, str):
                val = raw_val.strip().lower()
                if val in {"1", "true", "yes", "y", "on"}:
                    updates[key] = True
                elif val in {"0", "false", "no", "n", "off"}:
                    updates[key] = False
                else:
                    raise ValueError(f"Invalid boolean for {key}: {raw_val}")
            else:
                updates[key] = bool(raw_val)
            continue
        if isinstance(current_val, int):
            updates[key] = int(raw_val)
            continue
        if isinstance(current_val, float):
            updates[key] = float(raw_val)
            continue
        updates[key] = raw_val
    return normalize_control_config(replace(base, **updates))


def build_alternate_control_config(base: ControlConfig, profile: str) -> ControlConfig:
    normalized = str(profile or "").strip().lower()
    if normalized in ("", "none"):
        return base
    if normalized == "custom":
        return base
    if normalized == "runtime_optimized":
        return replace(
            base,
            min_context_coverage=max(0.28, base.min_context_coverage - 0.02),
            min_persona_coverage=max(0.15, base.min_persona_coverage - 0.02),
            relaxed_context_coverage=max(0.15, base.relaxed_context_coverage - 0.02),
            relaxed_persona_coverage=max(0.08, base.relaxed_persona_coverage - 0.01),
            relaxed_candidate_score=max(0.40, base.relaxed_candidate_score - 0.02),
            adaptive_candidate_score=max(0.34, base.adaptive_candidate_score - 0.02),
            adaptive_context_coverage=max(0.12, base.adaptive_context_coverage - 0.02),
            adaptive_persona_coverage=max(0.09, base.adaptive_persona_coverage - 0.01),
            rewrite_candidates=max(2, min(base.rewrite_candidates, 3)),
            adaptive_high_confidence_rewrites=1,
            adaptive_mid_confidence_rewrites=1,
            adaptive_low_confidence_rewrites=2,
            low_confidence_retry_requires_gain=True,
            low_confidence_retry_min_score_gain=max(0.015, base.low_confidence_retry_min_score_gain),
            low_confidence_retry_min_coverage_gain=max(0.03, base.low_confidence_retry_min_coverage_gain),
        )
    if normalized == "quality_optimized":
        return replace(
            base,
            min_context_coverage=min(0.42, base.min_context_coverage + 0.03),
            min_persona_coverage=min(0.24, base.min_persona_coverage + 0.02),
            relaxed_context_coverage=min(0.24, base.relaxed_context_coverage + 0.02),
            relaxed_persona_coverage=min(0.14, base.relaxed_persona_coverage + 0.01),
            relaxed_candidate_score=min(0.52, base.relaxed_candidate_score + 0.03),
            adaptive_candidate_score=min(0.47, base.adaptive_candidate_score + 0.04),
            adaptive_context_coverage=min(0.22, base.adaptive_context_coverage + 0.04),
            adaptive_persona_coverage=min(0.15, base.adaptive_persona_coverage + 0.03),
            rewrite_candidates=max(base.rewrite_candidates, 3),
            adaptive_high_confidence_rewrites=max(1, base.adaptive_high_confidence_rewrites),
            adaptive_mid_confidence_rewrites=max(2, base.adaptive_mid_confidence_rewrites),
            adaptive_low_confidence_rewrites=max(3, base.adaptive_low_confidence_rewrites),
            low_confidence_retry_requires_gain=True,
            low_confidence_retry_min_score_gain=min(0.01, base.low_confidence_retry_min_score_gain),
            low_confidence_retry_min_coverage_gain=min(0.02, base.low_confidence_retry_min_coverage_gain),
        )
    if normalized == "risk_latency_aware":
        return replace(
            base,
            intent_risk_adaptation_enabled=True,
            latency_adaptation_enabled=True,
            latency_relax_start_pressure=0.52,
            latency_relax_max_delta=0.10,
            low_risk_context_relax=max(base.low_risk_context_relax, 0.06),
            low_risk_persona_relax=max(base.low_risk_persona_relax, 0.04),
            low_risk_candidate_score_relax=max(base.low_risk_candidate_score_relax, 0.04),
            high_risk_context_tighten=max(base.high_risk_context_tighten, 0.05),
            high_risk_persona_tighten=max(base.high_risk_persona_tighten, 0.03),
            high_risk_candidate_score_tighten=max(base.high_risk_candidate_score_tighten, 0.04),
            adaptive_acceptance_enabled=True,
            allow_relaxed_acceptance=True,
            rewrite_candidates=max(2, min(base.rewrite_candidates, 3)),
            adaptive_high_confidence_rewrites=1,
            adaptive_mid_confidence_rewrites=max(1, min(base.adaptive_mid_confidence_rewrites, 2)),
            adaptive_low_confidence_rewrites=max(1, min(base.adaptive_low_confidence_rewrites, 2)),
        )
    if normalized == "hybrid_balanced":
        return replace(
            base,
            min_context_coverage=0.31,
            min_persona_coverage=0.17,
            relaxed_context_coverage=0.16,
            relaxed_persona_coverage=0.08,
            relaxed_candidate_score=0.42,
            adaptive_candidate_score=0.35,
            rewrite_candidates=2,
            adaptive_mid_confidence_rewrites=1,
            adaptive_low_confidence_rewrites=2,
            intent_risk_adaptation_enabled=True,
            latency_adaptation_enabled=True,
            latency_relax_start_pressure=0.50,
            latency_relax_max_delta=0.08,
            low_risk_context_relax=0.06,
            low_risk_persona_relax=0.04,
            low_risk_candidate_score_relax=0.04,
            high_risk_context_tighten=0.02,
            high_risk_persona_tighten=0.01,
            high_risk_candidate_score_tighten=0.01,
        )
    if normalized == "intent_focus_adaptive":
        return replace(
            base,
            min_context_coverage=max(0.30, base.min_context_coverage - 0.01),
            min_persona_coverage=max(0.16, base.min_persona_coverage - 0.01),
            relaxed_context_coverage=max(0.16, base.relaxed_context_coverage - 0.01),
            relaxed_persona_coverage=max(0.08, base.relaxed_persona_coverage - 0.005),
            relaxed_candidate_score=max(0.41, base.relaxed_candidate_score - 0.01),
            adaptive_candidate_score=max(0.34, base.adaptive_candidate_score - 0.01),
            rewrite_candidates=max(2, min(base.rewrite_candidates, 3)),
            adaptive_mid_confidence_rewrites=1,
            adaptive_low_confidence_rewrites=2,
            intent_risk_adaptation_enabled=True,
            latency_adaptation_enabled=True,
            latency_relax_start_pressure=0.52,
            latency_relax_max_delta=0.08,
            intent_focused_context_enabled=True,
            intent_focus_min_keep=3,
            intent_focus_keep_ratio_low=0.42,
            intent_focus_keep_ratio_medium=0.62,
            intent_focus_keep_ratio_high=0.95,
            intent_focus_min_relevance=0.18,
        )
    if normalized == "blend_balanced":
        return replace(
            base,
            min_context_coverage=0.31,
            min_persona_coverage=0.17,
            relaxed_context_coverage=0.16,
            relaxed_persona_coverage=0.08,
            relaxed_candidate_score=0.41,
            adaptive_candidate_score=0.34,
            adaptive_context_coverage=0.12,
            adaptive_persona_coverage=0.09,
            rewrite_candidates=2,
            adaptive_mid_confidence_rewrites=1,
            adaptive_low_confidence_rewrites=2,
            intent_risk_adaptation_enabled=True,
            latency_adaptation_enabled=True,
            latency_relax_start_pressure=0.52,
            latency_relax_max_delta=0.08,
            intent_focused_context_enabled=True,
            intent_focus_min_keep=3,
            intent_focus_keep_ratio_low=0.48,
            intent_focus_keep_ratio_medium=0.68,
            intent_focus_keep_ratio_high=1.0,
            intent_focus_min_relevance=0.20,
            near_pass_enabled=True,
            near_pass_max_context_gap=0.04,
            near_pass_max_persona_gap=0.03,
            near_pass_score_floor=0.35,
            near_pass_block_high_risk=True,
        )
    raise ValueError(f"Unsupported control-alt profile: {profile}")


def control_config_payload(
    config: ControlConfig,
    rewrite_budget_ms: float,
    rewrite_budget_multiplier: float,
) -> Dict[str, Any]:
    return {
        "min_context_coverage": config.min_context_coverage,
        "min_persona_coverage": config.min_persona_coverage,
        "min_response_tokens": config.min_response_tokens,
        "rewrite_temperature": config.rewrite_temperature,
        "rewrite_max_tokens": config.rewrite_max_tokens,
        "rewrite_candidates": config.rewrite_candidates,
        "rewrite_temperature_step": config.rewrite_temperature_step,
        "early_stop_on_pass": config.early_stop_on_pass,
        "early_stop_score": config.early_stop_score,
        "allow_relaxed_acceptance": config.allow_relaxed_acceptance,
        "relaxed_context_coverage": config.relaxed_context_coverage,
        "relaxed_persona_coverage": config.relaxed_persona_coverage,
        "relaxed_candidate_score": config.relaxed_candidate_score,
        "min_rewrite_gain": config.min_rewrite_gain,
        "enable_rewrite": config.enable_rewrite,
        "allow_best_effort_rewrite": config.allow_best_effort_rewrite,
        "behavior_adaptation_enabled": config.behavior_adaptation_enabled,
        "adaptive_acceptance_enabled": config.adaptive_acceptance_enabled,
        "adaptive_candidate_score": config.adaptive_candidate_score,
        "adaptive_context_coverage": config.adaptive_context_coverage,
        "adaptive_persona_coverage": config.adaptive_persona_coverage,
        "adaptive_high_confidence_score": config.adaptive_high_confidence_score,
        "adaptive_mid_confidence_score": config.adaptive_mid_confidence_score,
        "adaptive_high_confidence_rewrites": config.adaptive_high_confidence_rewrites,
        "adaptive_mid_confidence_rewrites": config.adaptive_mid_confidence_rewrites,
        "adaptive_low_confidence_rewrites": config.adaptive_low_confidence_rewrites,
        "low_confidence_retry_requires_gain": config.low_confidence_retry_requires_gain,
        "low_confidence_retry_min_score_gain": config.low_confidence_retry_min_score_gain,
        "low_confidence_retry_min_coverage_gain": config.low_confidence_retry_min_coverage_gain,
        "intent_risk_adaptation_enabled": config.intent_risk_adaptation_enabled,
        "latency_adaptation_enabled": config.latency_adaptation_enabled,
        "latency_relax_start_pressure": config.latency_relax_start_pressure,
        "latency_relax_max_delta": config.latency_relax_max_delta,
        "low_risk_context_relax": config.low_risk_context_relax,
        "low_risk_persona_relax": config.low_risk_persona_relax,
        "low_risk_candidate_score_relax": config.low_risk_candidate_score_relax,
        "high_risk_context_tighten": config.high_risk_context_tighten,
        "high_risk_persona_tighten": config.high_risk_persona_tighten,
        "high_risk_candidate_score_tighten": config.high_risk_candidate_score_tighten,
        "intent_focused_context_enabled": config.intent_focused_context_enabled,
        "intent_focus_min_keep": config.intent_focus_min_keep,
        "intent_focus_keep_ratio_low": config.intent_focus_keep_ratio_low,
        "intent_focus_keep_ratio_medium": config.intent_focus_keep_ratio_medium,
        "intent_focus_keep_ratio_high": config.intent_focus_keep_ratio_high,
        "intent_focus_min_relevance": config.intent_focus_min_relevance,
        "near_pass_enabled": config.near_pass_enabled,
        "near_pass_max_context_gap": config.near_pass_max_context_gap,
        "near_pass_max_persona_gap": config.near_pass_max_persona_gap,
        "near_pass_score_floor": config.near_pass_score_floor,
        "near_pass_block_high_risk": config.near_pass_block_high_risk,
        "rewrite_budget_ms": rewrite_budget_ms,
        "rewrite_budget_multiplier": rewrite_budget_multiplier,
    }


def metric_names(include_bertscore: bool) -> List[str]:
    names = list(BASE_METRICS) + list(GAMEPLAY_METRICS) + list(DIAGNOSTIC_METRICS)
    if include_bertscore:
        names.append("bertscore_f1")
    names.append("overall_quality")
    return names


def quality_weights(has_bertscore: bool) -> Dict[str, float]:
    if has_bertscore:
        return {
            "context_relevance": 0.30,
            "persona_consistency": 0.27,
            "naturalness": 0.18,
            "bertscore_f1": 0.12,
            "context_keyword_coverage": 0.08,
            "persona_keyword_coverage": 0.05,
        }
    return {
        "context_relevance": 0.36,
        "persona_consistency": 0.30,
        "naturalness": 0.20,
        "context_keyword_coverage": 0.08,
        "persona_keyword_coverage": 0.06,
    }


def row_overall_quality(row: Dict[str, Any]) -> float:
    has_bertscore = "bertscore_f1" in row and not math.isnan(float(row.get("bertscore_f1", float("nan"))))
    weights = quality_weights(has_bertscore=has_bertscore)
    total = 0.0
    denom = 0.0
    for metric, weight in weights.items():
        if metric not in row:
            continue
        val = float(row.get(metric, 0.0))
        if math.isnan(val):
            continue
        total += weight * val
        denom += weight
    return (total / denom) if denom > 0.0 else 0.0


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    # Use utf-8-sig to tolerate BOM-prefixed files generated by some editors/shells.
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return float("nan")
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = (len(sorted_vals) - 1) * p
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return sorted_vals[lo]
    frac = idx - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def bootstrap_mean_ci(values: Sequence[float], seed: int, iters: int = 2000) -> Dict[str, float]:
    if not values:
        return {
            "n": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p95": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
        }

    vals = [float(v) for v in values]
    mean_val = statistics.fmean(vals)
    median_val = statistics.median(vals)
    p95_val = percentile(vals, 0.95)
    n = len(vals)
    if n == 1:
        low = high = mean_val
    else:
        rng = random.Random(seed)
        samples: List[float] = []
        for _ in range(iters):
            draw = [vals[rng.randrange(n)] for __ in range(n)]
            samples.append(statistics.fmean(draw))
        low = percentile(samples, 0.025)
        high = percentile(samples, 0.975)

    return {
        "n": n,
        "mean": mean_val,
        "median": median_val,
        "p95": p95_val,
        "ci95_low": low,
        "ci95_high": high,
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
    mean_delta = statistics.fmean(deltas)
    if n == 1:
        return {
            "n": 1,
            "mean_delta": mean_delta,
            "ci95_low": mean_delta,
            "ci95_high": mean_delta,
            "p_delta_le_0": 1.0 if mean_delta <= 0 else 0.0,
        }

    rng = random.Random(seed)
    sampled_means: List[float] = []
    for _ in range(iters):
        draw = [deltas[rng.randrange(n)] for __ in range(n)]
        sampled_means.append(statistics.fmean(draw))
    ci95_low = percentile(sampled_means, 0.025)
    ci95_high = percentile(sampled_means, 0.975)
    p_delta_le_0 = sum(1 for x in sampled_means if x <= 0.0) / float(len(sampled_means))
    return {
        "n": n,
        "mean_delta": mean_delta,
        "ci95_low": ci95_low,
        "ci95_high": ci95_high,
        "p_delta_le_0": p_delta_le_0,
    }


def paired_cluster_bootstrap_delta_ci(
    deltas_by_cluster: Dict[str, List[float]],
    seed: int,
    iters: int = 3000,
) -> Dict[str, float]:
    cluster_ids = sorted([cid for cid, vals in deltas_by_cluster.items() if vals])
    n_clusters = len(cluster_ids)
    n_pairs = sum(len(deltas_by_cluster[cid]) for cid in cluster_ids)
    if n_clusters == 0 or n_pairs == 0:
        return {
            "n_clusters": 0,
            "n_pairs": 0,
            "mean_delta": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "p_delta_le_0": float("nan"),
        }

    all_vals: List[float] = []
    for cid in cluster_ids:
        all_vals.extend(deltas_by_cluster[cid])
    mean_delta = statistics.fmean(all_vals)
    if n_clusters == 1:
        return {
            "n_clusters": 1,
            "n_pairs": n_pairs,
            "mean_delta": mean_delta,
            "ci95_low": mean_delta,
            "ci95_high": mean_delta,
            "p_delta_le_0": 1.0 if mean_delta <= 0 else 0.0,
        }

    rng = random.Random(seed)
    sampled_means: List[float] = []
    for _ in range(iters):
        sampled_vals: List[float] = []
        for __ in range(n_clusters):
            cid = cluster_ids[rng.randrange(n_clusters)]
            sampled_vals.extend(deltas_by_cluster[cid])
        if sampled_vals:
            sampled_means.append(statistics.fmean(sampled_vals))
    if not sampled_means:
        sampled_means = [mean_delta]
    ci95_low = percentile(sampled_means, 0.025)
    ci95_high = percentile(sampled_means, 0.975)
    p_delta_le_0 = sum(1 for x in sampled_means if x <= 0.0) / float(len(sampled_means))
    return {
        "n_clusters": n_clusters,
        "n_pairs": n_pairs,
        "mean_delta": mean_delta,
        "ci95_low": ci95_low,
        "ci95_high": ci95_high,
        "p_delta_le_0": p_delta_le_0,
    }


def safe_shell(command: Sequence[str], timeout_s: int = 15) -> str:
    try:
        result = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except Exception:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def gather_hardware_metadata() -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "timestamp_utc": utc_iso(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "cpu_count_logical": os.cpu_count(),
    }

    if platform.system().lower().startswith("win"):
        cpu_name = safe_shell(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name).Trim()",
            ]
        )
        ram_bytes = safe_shell(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory",
            ]
        )
        if cpu_name:
            metadata["cpu_name"] = cpu_name
        if ram_bytes.isdigit():
            metadata["ram_bytes"] = int(ram_bytes)
            metadata["ram_gb"] = round(int(ram_bytes) / (1024**3), 2)
    else:
        cpu_name = safe_shell(["sh", "-lc", "cat /proc/cpuinfo | grep 'model name' | head -n 1 | cut -d: -f2"])
        mem_total = safe_shell(["sh", "-lc", "cat /proc/meminfo | grep MemTotal | awk '{print $2}'"])
        if cpu_name:
            metadata["cpu_name"] = cpu_name.strip()
        if mem_total.isdigit():
            metadata["ram_kb"] = int(mem_total)
            metadata["ram_gb"] = round(int(mem_total) / (1024**2), 2)

    gpu_csv = safe_shell(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        timeout_s=10,
    )
    metadata["gpu"] = []
    if gpu_csv:
        for line in gpu_csv.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                metadata["gpu"].append(
                    {
                        "name": parts[0],
                        "memory_total": parts[1],
                        "driver_version": parts[2],
                    }
                )
    return metadata


def gather_code_revision_metadata(root_dir: Path) -> Dict[str, Any]:
    root = str(root_dir.resolve())
    commit = safe_shell(["git", "-C", root, "rev-parse", "HEAD"])
    short_commit = safe_shell(["git", "-C", root, "rev-parse", "--short", "HEAD"])
    branch = safe_shell(["git", "-C", root, "rev-parse", "--abbrev-ref", "HEAD"])
    status = safe_shell(["git", "-C", root, "status", "--porcelain"])
    metadata: Dict[str, Any] = {
        "git_available": bool(commit),
        "commit": commit,
        "short_commit": short_commit,
        "branch": branch,
        "dirty": bool(status.strip()),
    }
    if status.strip():
        lines = [line.strip() for line in status.splitlines() if line.strip()]
        metadata["dirty_file_count"] = len(lines)
        metadata["dirty_file_preview"] = lines[:25]
    return metadata


def collect_tracked_file_hashes(root_dir: Path, rel_paths: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for rel in rel_paths:
        path = (root_dir / rel).resolve()
        if not path.exists():
            out[rel] = {"exists": False}
            continue
        out[rel] = {
            "exists": True,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    return out


def query_ollama_model(host: str, model: str) -> Dict[str, Any]:
    for payload in ({"name": model}, {"model": model}):
        try:
            resp = requests.post(f"{host}/api/show", json=payload, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                tensors = data.get("tensors")
                if isinstance(tensors, list):
                    data["tensor_count"] = len(tensors)
                    data["tensor_preview"] = tensors[:8]
                    data.pop("tensors", None)
                data["model_name"] = model
                data["host"] = host
                return data
        except Exception as exc:
            return {"model_name": model, "host": host, "error": str(exc)}
    return {"model_name": model, "host": host, "error": "Failed to query /api/show"}


def build_prompt(row: Dict[str, Any], include_dynamic_context: bool) -> str:
    persona = str(row.get("persona", "")).strip()
    context = str(row.get("dynamic_context", "")).strip()
    player_input = str(row.get("player_input", "")).strip()

    lines: List[str] = []
    lines.append("You are roleplaying one NPC in a narrative game.")
    lines.append(f"Persona: {persona}")
    lines.append("Rules:")
    lines.append("- Stay in persona.")
    lines.append("- Write 2-3 natural sentences.")
    lines.append("- Return only spoken dialogue, no labels, no metadata, no templates.")
    lines.append("- Never output hidden fields such as Temporal Memories or BehaviorTreeState.")
    if include_dynamic_context:
        lines.append("Current dynamic game state:")
        for chunk in [c.strip() for c in context.split(";") if c.strip()]:
            lines.append(f"- {chunk.replace('=', ': ')}")
        lines.append("Use at least two concrete details from current dynamic game state.")
    else:
        lines.append("Current dynamic game state is unavailable.")
    lines.append(f"Player says: {player_input}")
    lines.append("NPC reply:")
    return "\n".join(lines)


def sanitize_npc_response(text: str) -> str:
    return sanitize_response(text)


def generate_ollama_response(
    host: str,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout_s: int,
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    try:
        resp = requests.post(f"{host}/api/generate", json=payload, timeout=timeout_s)
    except Exception as exc:
        return {"ok": False, "error": f"request_failed: {exc}"}
    if resp.status_code != 200:
        return {
            "ok": False,
            "error": f"http_{resp.status_code}",
            "body": resp.text[:220],
        }
    try:
        data = resp.json()
    except Exception as exc:
        return {"ok": False, "error": f"json_decode_failed: {exc}"}

    response_text = str(data.get("response", "")).strip()
    return {
        "ok": True,
        "response": response_text,
        "eval_count": int(data.get("eval_count", 0) or 0),
        "prompt_eval_count": int(data.get("prompt_eval_count", 0) or 0),
        "eval_duration_ns": int(data.get("eval_duration", 0) or 0),
        "total_duration_ns": int(data.get("total_duration", 0) or 0),
    }


def keyword_coverage(text: str, keywords: Sequence[str]) -> float:
    kws = [str(k).strip().lower() for k in keywords if str(k).strip()]
    if not kws:
        return 0.0
    lowered = text.lower()
    text_tokens = set(tokenize(text))
    hits = 0
    for kw in kws:
        if kw in lowered:
            hits += 1
            continue
        kw_tokens = tokenize(kw)
        if kw_tokens and all(token in text_tokens for token in kw_tokens):
            hits += 1
    return hits / len(kws)


def jaccard_overlap(a: str, b: str) -> float:
    ta = set(tokenize(a))
    tb = set(tokenize(b))
    if not ta or not tb:
        return 0.0
    return len(ta.intersection(tb)) / len(ta.union(tb))


def repetition_ratio(text: str, n: int = 3) -> float:
    tokens = tokenize(text)
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[idx : idx + n]) for idx in range(len(tokens) - n + 1)]
    if not grams:
        return 0.0
    counts: Dict[tuple[str, ...], int] = {}
    for gram in grams:
        counts[gram] = counts.get(gram, 0) + 1
    repeats = sum(count - 1 for count in counts.values() if count > 1)
    return repeats / len(grams)


def sentence_count(text: str) -> int:
    rough = [x.strip() for x in re.split(r"[.!?]+", text) if x.strip()]
    return len(rough)


def style_score(persona: str, response: str) -> float:
    persona_l = persona.lower()
    words = tokenize(response)
    wc = len(words)
    scores: List[float] = []

    if "brief" in persona_l or "short" in persona_l:
        if wc <= 55:
            scores.append(1.0)
        else:
            scores.append(max(0.0, 1.0 - (wc - 55) / 80.0))
    if "talkative" in persona_l:
        scores.append(min(1.0, wc / 32.0))
    if "formal" in persona_l:
        long_words = sum(1 for w in words if len(w) >= 7)
        scores.append(min(1.0, long_words / max(1.0, wc * 0.25)))
    if "mysterious" in persona_l or "indirect" in persona_l:
        marker_hit = keyword_coverage(response, ["perhaps", "price", "ritual", "shadow", "moon", "curse"])
        scores.append(marker_hit)
    if "procedural" in persona_l:
        marker_hit = keyword_coverage(response, ["statement", "verify", "evidence", "process"])
        scores.append(marker_hit)

    if not scores:
        return 0.5
    return float(sum(scores) / len(scores))


def context_relevance_score(response: str, context: str, context_keywords: Sequence[str]) -> float:
    kw = keyword_coverage(response, context_keywords)
    overlap = jaccard_overlap(response, context)
    return 0.7 * kw + 0.3 * overlap


def persona_consistency_score(response: str, persona: str, persona_keywords: Sequence[str]) -> float:
    kw = keyword_coverage(response, persona_keywords)
    sty = style_score(persona, response)
    return 0.8 * kw + 0.2 * sty


def naturalness_components(response: str) -> Dict[str, float]:
    words = tokenize(response)
    wc = len(words)
    if wc == 0:
        return {
            "naturalness": 0.0,
            "distinct1": 0.0,
            "length_score": 0.0,
            "sentence_score": 0.0,
        }

    distinct_1 = len(set(words)) / wc
    rep = repetition_ratio(response, n=3)

    target = 38.0
    spread = 30.0
    length_score = max(0.0, 1.0 - abs(wc - target) / spread)

    sc = sentence_count(response)
    if 2 <= sc <= 3:
        sentence_score = 1.0
    elif sc == 1 or sc == 4:
        sentence_score = 0.65
    else:
        sentence_score = 0.35

    score = 0.4 * distinct_1 + 0.3 * (1.0 - rep) + 0.2 * length_score + 0.1 * sentence_score
    return {
        "naturalness": max(0.0, min(1.0, score)),
        "distinct1": max(0.0, min(1.0, distinct_1)),
        "length_score": max(0.0, min(1.0, length_score)),
        "sentence_score": max(0.0, min(1.0, sentence_score)),
    }


def naturalness_score(response: str) -> float:
    return float(naturalness_components(response).get("naturalness", 0.0))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _event_keywords(text: str, max_items: int = 12) -> List[str]:
    tokens = [t for t in tokenize(text) if len(t) >= 4]
    if not tokens:
        return []
    deduped: List[str] = []
    seen = set()
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        deduped.append(token)
        if len(deduped) >= max_items:
            break
    return deduped


def _urgency_flag(player_input: str, recent_event: str) -> bool:
    merged = f"{player_input} {recent_event}".lower()
    urgency_markers = (
        "urgent",
        "immediate",
        "quick",
        "quickly",
        "right now",
        "running out of time",
        "priority",
        "critical",
        "no delays",
        "asap",
    )
    return any(marker in merged for marker in urgency_markers)


def _detect_policy_stance(response: str) -> str:
    lowered = response.lower()
    allow_markers = (
        "you may",
        "go ahead",
        "entry granted",
        "i allow",
        "authorized",
        "permit granted",
        "pass through",
    )
    deny_markers = (
        "cannot",
        "can't",
        "not allowed",
        "no entry",
        "denied",
        "refuse",
        "no passage",
        "i won't allow",
    )
    allow = any(marker in lowered for marker in allow_markers)
    deny = any(marker in lowered for marker in deny_markers)
    if allow and deny:
        return "mixed"
    if allow:
        return "allow"
    if deny:
        return "deny"
    return "neutral"


def quest_state_correctness_score(
    response: str,
    scenario: Dict[str, Any],
    context_relevance: float,
    context_keyword_cov: float,
) -> float:
    tags = get_scenario_tags(scenario)
    conflict_type = str(tags.get("conflict_type", "")).strip().lower()
    reference = str(scenario.get("reference_response", ""))
    ref_overlap = jaccard_overlap(response, reference)
    ctx_map = parse_dynamic_context_map(str(scenario.get("dynamic_context", "")))
    state_terms = [
        str(ctx_map.get("behaviortreestate", "")).strip(),
        str(ctx_map.get("location", "")).strip(),
        str(ctx_map.get("recentevent", "")).strip(),
    ]
    state_terms.extend(_event_keywords(str(ctx_map.get("recentevent", ""))))
    state_cov = keyword_coverage(response, [x for x in state_terms if x])

    policy_alignment = 0.5
    if conflict_type == "access_control":
        response_stance = _detect_policy_stance(response)
        reference_stance = _detect_policy_stance(reference)
        if response_stance == "mixed":
            policy_alignment = 0.0
        elif reference_stance == "neutral" and response_stance != "mixed":
            policy_alignment = 0.6
        elif response_stance == reference_stance:
            policy_alignment = 1.0
        elif response_stance == "neutral":
            policy_alignment = 0.45
        else:
            policy_alignment = 0.1

    return clamp01(
        0.36 * float(context_relevance)
        + 0.22 * float(context_keyword_cov)
        + 0.22 * float(state_cov)
        + 0.20 * max(float(ref_overlap), float(policy_alignment))
    )


def lore_consistency_score(response: str, scenario: Dict[str, Any], context_keyword_cov: float) -> float:
    context = str(scenario.get("dynamic_context", ""))
    reference = str(scenario.get("reference_response", ""))
    ref_overlap = jaccard_overlap(response, reference)

    context_entities = set(re.findall(r"\b[A-Z][a-z]{2,}\b", context + " " + reference))
    response_entities = set(re.findall(r"\b[A-Z][a-z]{2,}\b", response))
    if context_entities:
        entity_cov = len(response_entities.intersection(context_entities)) / len(context_entities)
    else:
        entity_cov = 0.5
    hallucinated_entities = [x for x in response_entities if x not in context_entities]
    hallucination_penalty = min(0.25, 0.05 * len(hallucinated_entities))

    return clamp01(0.48 * float(context_keyword_cov) + 0.32 * float(ref_overlap) + 0.20 * float(entity_cov) - hallucination_penalty)


def objective_completion_support_score(response: str, scenario: Dict[str, Any]) -> float:
    tags = get_scenario_tags(scenario)
    conflict_type = str(tags.get("conflict_type", "")).strip().lower()
    actionability = keyword_coverage(
        response,
        [
            "first",
            "then",
            "next",
            "go to",
            "bring",
            "show",
            "speak to",
            "return with",
            "verify",
            "confirm",
        ],
    )
    procedural = keyword_coverage(response, ["if", "when", "once", "after", "before", "until"])
    has_clear_outcome = keyword_coverage(response, ["you can", "you may", "i will", "that will", "this will"])

    if conflict_type == "access_control":
        stance = _detect_policy_stance(response)
        stance_bonus = 1.0 if stance in ("allow", "deny") else (0.25 if stance == "neutral" else 0.0)
        return clamp01(0.45 * actionability + 0.25 * procedural + 0.15 * has_clear_outcome + 0.15 * stance_bonus)

    return clamp01(0.55 * actionability + 0.25 * procedural + 0.20 * has_clear_outcome)


def gameplay_usefulness_score(
    response: str,
    objective_completion_support: float,
    context_relevance: float,
    naturalness: float,
) -> float:
    wc = len(tokenize(response))
    concise_score = clamp01(1.0 - abs(float(wc) - 34.0) / 40.0)
    return clamp01(
        0.42 * float(objective_completion_support)
        + 0.28 * float(context_relevance)
        + 0.20 * float(naturalness)
        + 0.10 * float(concise_score)
    )


def time_pressure_acceptability_score(
    response: str,
    scenario: Dict[str, Any],
    objective_completion_support: float,
) -> float:
    ctx_map = parse_dynamic_context_map(str(scenario.get("dynamic_context", "")))
    is_urgent = _urgency_flag(str(scenario.get("player_input", "")), str(ctx_map.get("recentevent", "")))
    wc = len(tokenize(response))
    directness = keyword_coverage(response, ["now", "immediately", "first", "then", "go", "show", "do this"])
    concise = clamp01(1.0 - max(0.0, float(wc) - 45.0) / 55.0)
    if is_urgent:
        return clamp01(0.58 * float(objective_completion_support) + 0.27 * float(concise) + 0.15 * float(directness))
    return clamp01(0.45 * float(objective_completion_support) + 0.25 * float(concise) + 0.30 * float(directness))


def compute_optional_bertscore(
    predictions: List[str],
    references: List[str],
    lang: str = "en",
    model_type: str = "",
    batch_size: int = 16,
    cache_dir: str = "",
) -> Dict[str, Any]:
    try:
        import transformers.utils.import_utils as iu  # type: ignore
        iu._torchao_available = False
        iu._torchao_version = "0.0.0"
    except Exception:
        pass

    try:
        from bert_score import score as bert_score  # type: ignore
    except Exception as exc:
        return {"available": False, "reason": str(exc)}

    env_backup: Dict[str, Optional[str]] = {}
    resolved_cache_dir = str(cache_dir).strip()
    if resolved_cache_dir:
        cache_path = Path(resolved_cache_dir).expanduser()
        cache_path.mkdir(parents=True, exist_ok=True)
        resolved_cache_dir = str(cache_path)
        for key in ("HF_HOME", "TRANSFORMERS_CACHE", "HUGGINGFACE_HUB_CACHE"):
            env_backup[key] = os.environ.get(key)
            os.environ[key] = resolved_cache_dir

    kwargs: Dict[str, Any] = {
        "lang": lang,
        "rescale_with_baseline": True,
        "batch_size": max(1, int(batch_size)),
    }
    chosen_model = str(model_type).strip()
    if chosen_model:
        kwargs["model_type"] = chosen_model

    try:
        _, _, f1 = bert_score(predictions, references, **kwargs)
        values = [float(x) for x in f1]
        return {
            "available": True,
            "values": values,
            "lang": lang,
            "model_type": chosen_model or "auto",
            "batch_size": int(kwargs["batch_size"]),
            "cache_dir": resolved_cache_dir,
        }
    except Exception as exc:
        return {"available": False, "reason": str(exc)}
    finally:
        if env_backup:
            for key, value in env_backup.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def evaluate_responses_for_arm(
    responses: List[Dict[str, Any]],
    scenarios_by_id: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    scored: List[Dict[str, Any]] = []
    source_stances: Dict[str, set[str]] = {}
    source_id_by_scenario: Dict[str, str] = {}
    pending_rows: List[Dict[str, Any]] = []
    for row in responses:
        sid = str(row.get("scenario_id", ""))
        scenario = scenarios_by_id.get(sid, {})
        response = str(row.get("response", ""))
        persona = str(scenario.get("persona", ""))
        context = str(scenario.get("dynamic_context", ""))
        context_keywords = [str(x) for x in scenario.get("context_keywords", [])]
        persona_keywords = [str(x) for x in scenario.get("persona_keywords", [])]

        context_kw = keyword_coverage(response, context_keywords)
        context_ov = jaccard_overlap(response, context)
        persona_kw = keyword_coverage(response, persona_keywords)
        persona_sty = style_score(persona, response)
        nat = naturalness_components(response)

        context_rel = context_relevance_score(response, context, context_keywords)
        persona_cons = persona_consistency_score(response, persona, persona_keywords)
        natural = float(nat.get("naturalness", 0.0))

        quest_state_correctness = quest_state_correctness_score(
            response=response,
            scenario=scenario,
            context_relevance=context_rel,
            context_keyword_cov=context_kw,
        )
        lore_consistency = lore_consistency_score(
            response=response,
            scenario=scenario,
            context_keyword_cov=context_kw,
        )
        objective_completion_support = objective_completion_support_score(response=response, scenario=scenario)
        gameplay_usefulness = gameplay_usefulness_score(
            response=response,
            objective_completion_support=objective_completion_support,
            context_relevance=context_rel,
            naturalness=natural,
        )
        time_pressure_acceptability = time_pressure_acceptability_score(
            response=response,
            scenario=scenario,
            objective_completion_support=objective_completion_support,
        )

        source_scenario_id = str(scenario.get("source_scenario_id", "")).strip() or sid
        source_id_by_scenario[sid] = source_scenario_id
        stance = _detect_policy_stance(response)
        if stance != "neutral":
            source_stances.setdefault(source_scenario_id, set()).add(stance)

        pending_rows.append(
            {
                **row,
                "context_relevance": context_rel,
                "persona_consistency": persona_cons,
                "naturalness": natural,
                "quest_state_correctness": quest_state_correctness,
                "lore_consistency": lore_consistency,
                "objective_completion_support": objective_completion_support,
                "gameplay_usefulness": gameplay_usefulness,
                "time_pressure_acceptability": time_pressure_acceptability,
                "context_keyword_coverage": context_kw,
                "context_overlap": context_ov,
                "persona_keyword_coverage": persona_kw,
                "persona_style": persona_sty,
                "distinct1": float(nat.get("distinct1", 0.0)),
                "length_score": float(nat.get("length_score", 0.0)),
                "sentence_score": float(nat.get("sentence_score", 0.0)),
            }
        )

    source_contradiction_rate: Dict[str, float] = {}
    for source_id, stances in source_stances.items():
        source_contradiction_rate[source_id] = 1.0 if len(stances) > 1 else 0.0

    for row in pending_rows:
        sid = str(row.get("scenario_id", ""))
        source_id = source_id_by_scenario.get(sid, sid)
        contradiction_rate = float(source_contradiction_rate.get(source_id, 0.0))
        row["multi_turn_contradiction_rate"] = contradiction_rate
        row["multi_turn_contradiction_safety"] = 1.0 - contradiction_rate
        scored.append(row)
    return scored


def summarize_arm_scores(
    scored_rows: List[Dict[str, Any]],
    seed: int,
    include_bertscore: bool,
) -> Dict[str, Any]:
    metrics = list(BASE_METRICS) + list(GAMEPLAY_METRICS) + list(DIAGNOSTIC_METRICS)
    if include_bertscore:
        metrics.append("bertscore_f1")

    out: Dict[str, Any] = {"sample_count": len(scored_rows)}
    for idx, metric in enumerate(metrics):
        vals = [float(row.get(metric, float("nan"))) for row in scored_rows]
        vals = [x for x in vals if not math.isnan(x)]
        out[metric] = bootstrap_mean_ci(vals, seed=seed + 17 * (idx + 1))

    overall_vals: List[float] = []
    for row in scored_rows:
        if "overall_quality" in row:
            overall_vals.append(float(row.get("overall_quality", 0.0)))
        else:
            overall_vals.append(row_overall_quality(row))
    out["overall_quality"] = bootstrap_mean_ci(overall_vals, seed=seed + 211)
    out["metrics"] = metrics + ["overall_quality"]

    return out


def diff_against_reference(
    summary: Dict[str, Dict[str, Any]],
    target_arm: str,
    baseline_arm: str,
    metrics: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    metric_list = list(metrics) if metrics else metric_names(include_bertscore=True)
    out: Dict[str, Any] = {}
    target = summary.get(target_arm, {})
    baseline = summary.get(baseline_arm, {})
    for metric in metric_list:
        target_mean = target.get(metric, {}).get("mean")
        base_mean = baseline.get(metric, {}).get("mean")
        if target_mean is None or base_mean is None:
            continue
        if math.isnan(float(target_mean)) or math.isnan(float(base_mean)):
            continue
        abs_delta = float(target_mean) - float(base_mean)
        rel_delta = (abs_delta / float(base_mean)) if float(base_mean) != 0 else float("nan")
        out[metric] = {
            "target_mean": float(target_mean),
            "baseline_mean": float(base_mean),
            "absolute_delta": abs_delta,
            "relative_delta": rel_delta,
        }
    return out


def paired_metric_deltas(
    scored_by_arm: Dict[str, List[Dict[str, Any]]],
    target_arm: str,
    baseline_arm: str,
    seed: int,
    metrics: Optional[Sequence[str]] = None,
    scenarios_by_id: Optional[Dict[str, Dict[str, Any]]] = None,
    cluster_field: str = "source_scenario_id",
) -> Dict[str, Any]:
    metric_list = list(metrics) if metrics else metric_names(include_bertscore=True)
    target_rows = scored_by_arm.get(target_arm, [])
    baseline_rows = scored_by_arm.get(baseline_arm, [])
    if not target_rows or not baseline_rows:
        return {}

    target_map: Dict[tuple[str, int], Dict[str, Any]] = {}
    baseline_map: Dict[tuple[str, int], Dict[str, Any]] = {}
    for row in target_rows:
        key = (str(row.get("scenario_id", "")), int(row.get("repeat_index", 0) or 0))
        target_map[key] = row
    for row in baseline_rows:
        key = (str(row.get("scenario_id", "")), int(row.get("repeat_index", 0) or 0))
        baseline_map[key] = row

    common_keys = sorted(set(target_map.keys()).intersection(set(baseline_map.keys())))
    if not common_keys:
        return {}

    out: Dict[str, Any] = {}
    for metric_idx, metric in enumerate(metric_list):
        target_vals: List[float] = []
        baseline_vals: List[float] = []
        metric_keys: List[tuple[str, int]] = []
        for key in common_keys:
            t_row = target_map[key]
            b_row = baseline_map[key]
            if metric == "overall_quality":
                t_val = float(t_row.get("overall_quality", row_overall_quality(t_row)))
                b_val = float(b_row.get("overall_quality", row_overall_quality(b_row)))
            else:
                if metric not in t_row or metric not in b_row:
                    continue
                t_val = float(t_row.get(metric, 0.0))
                b_val = float(b_row.get(metric, 0.0))
            if math.isnan(t_val) or math.isnan(b_val):
                continue
            target_vals.append(t_val)
            baseline_vals.append(b_val)
            metric_keys.append(key)

        if not target_vals or not baseline_vals:
            continue
        metric_payload: Dict[str, Any] = paired_bootstrap_delta_ci(
            target_values=target_vals,
            baseline_values=baseline_vals,
            seed=seed + 101 * (metric_idx + 1),
        )
        if scenarios_by_id:
            deltas_by_cluster: Dict[str, List[float]] = {}
            for idx, key in enumerate(metric_keys):
                sid = str(key[0])
                scenario = scenarios_by_id.get(sid, {})
                cluster_id = str(scenario.get(cluster_field, "")).strip() or sid
                deltas_by_cluster.setdefault(cluster_id, []).append(float(target_vals[idx] - baseline_vals[idx]))
            metric_payload["cluster_bootstrap"] = paired_cluster_bootstrap_delta_ci(
                deltas_by_cluster=deltas_by_cluster,
                seed=seed + 10001 + 131 * (metric_idx + 1),
            )
            metric_payload["cluster_field"] = cluster_field
        out[metric] = metric_payload
    return out


def paired_metric_win_rates(
    scored_by_arm: Dict[str, List[Dict[str, Any]]],
    target_arm: str,
    baseline_arm: str,
    seed: int,
    metrics: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    metric_list = list(metrics) if metrics else metric_names(include_bertscore=True)
    target_rows = scored_by_arm.get(target_arm, [])
    baseline_rows = scored_by_arm.get(baseline_arm, [])
    if not target_rows or not baseline_rows:
        return {}

    target_map: Dict[tuple[str, int], Dict[str, Any]] = {}
    baseline_map: Dict[tuple[str, int], Dict[str, Any]] = {}
    for row in target_rows:
        key = (str(row.get("scenario_id", "")), int(row.get("repeat_index", 0) or 0))
        target_map[key] = row
    for row in baseline_rows:
        key = (str(row.get("scenario_id", "")), int(row.get("repeat_index", 0) or 0))
        baseline_map[key] = row
    common_keys = sorted(set(target_map.keys()).intersection(set(baseline_map.keys())))
    if not common_keys:
        return {}

    out: Dict[str, Any] = {}
    for metric_idx, metric in enumerate(metric_list):
        wins = 0
        losses = 0
        ties = 0
        soft_outcomes: List[float] = []
        strict_outcomes: List[float] = []
        for key in common_keys:
            t_row = target_map[key]
            b_row = baseline_map[key]
            if metric == "overall_quality":
                t_val = float(t_row.get("overall_quality", row_overall_quality(t_row)))
                b_val = float(b_row.get("overall_quality", row_overall_quality(b_row)))
            else:
                if metric not in t_row or metric not in b_row:
                    continue
                t_val = float(t_row.get(metric, float("nan")))
                b_val = float(b_row.get(metric, float("nan")))
            if math.isnan(t_val) or math.isnan(b_val):
                continue

            if t_val > b_val:
                wins += 1
                soft_outcomes.append(1.0)
                strict_outcomes.append(1.0)
            elif t_val < b_val:
                losses += 1
                soft_outcomes.append(0.0)
                strict_outcomes.append(0.0)
            else:
                ties += 1
                soft_outcomes.append(0.5)

        compared_n = wins + losses + ties
        if compared_n == 0:
            continue

        soft_summary = bootstrap_mean_ci(soft_outcomes, seed=seed + 131 * (metric_idx + 1))
        non_tie_n = wins + losses
        if non_tie_n > 0:
            strict_summary = bootstrap_mean_ci(strict_outcomes, seed=seed + 151 * (metric_idx + 1))
        else:
            strict_summary = {
                "n": 0,
                "mean": float("nan"),
                "median": float("nan"),
                "p95": float("nan"),
                "ci95_low": float("nan"),
                "ci95_high": float("nan"),
            }

        out[metric] = {
            "n": compared_n,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "soft_win_rate": soft_summary,
            "strict_non_tie_win_rate": strict_summary,
        }
    return out


def parse_dynamic_context_map(dynamic_context: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for chunk in [x.strip() for x in str(dynamic_context).split(";") if x.strip()]:
        if "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        out[str(key).strip().lower()] = str(value).strip()
    return out


def infer_location_type(location: str) -> str:
    lowered = location.lower()
    if any(k in lowered for k in ("gate", "wall", "checkpoint")):
        return "security"
    if any(k in lowered for k in ("market", "shop", "stall")):
        return "commerce"
    if any(k in lowered for k in ("library", "archive", "school")):
        return "knowledge"
    if any(k in lowered for k in ("forge", "smith", "workshop")):
        return "craft"
    if any(k in lowered for k in ("dungeon", "cell", "prison")):
        return "detention"
    if any(k in lowered for k in ("forest", "grove", "wild")):
        return "wilderness"
    if any(k in lowered for k in ("clinic", "hut", "healer", "hospital")):
        return "medical"
    return "other"


def infer_conflict_type(player_input: str) -> str:
    lowered = player_input.lower()
    if any(k in lowered for k in ("discount", "price", "cheap", "bargain")):
        return "economic_negotiation"
    if any(k in lowered for k in ("innocent", "trust", "prove", "why are you")):
        return "credibility_dispute"
    if any(k in lowered for k in ("help", "cure", "heal")):
        return "assistance_request"
    if any(k in lowered for k in ("weapon", "how should", "what should")):
        return "advisory_request"
    if any(k in lowered for k in ("let me", "entry", "open", "allow")):
        return "access_control"
    return "general_dialogue"


def infer_persona_archetype(persona: str) -> str:
    lowered = persona.lower()
    candidates = (
        "gatekeeper",
        "merchant",
        "healer",
        "blacksmith",
        "scholar",
        "captain",
        "prisoner",
        "witch",
    )
    for name in candidates:
        if name in lowered:
            return name
    return "other"


def get_scenario_tags(scenario: Dict[str, Any]) -> Dict[str, str]:
    tags: Dict[str, str] = {}
    raw_tags = scenario.get("scenario_tags")
    if isinstance(raw_tags, dict):
        for key, value in raw_tags.items():
            key_s = str(key).strip()
            val_s = str(value).strip()
            if key_s and val_s:
                tags[key_s] = val_s

    ctx = parse_dynamic_context_map(str(scenario.get("dynamic_context", "")))
    if "behavior_state" not in tags and ctx.get("behaviortreestate"):
        tags["behavior_state"] = str(ctx.get("behaviortreestate", "")).strip()
    if "location_type" not in tags:
        loc = str(ctx.get("location", ""))
        if loc:
            tags["location_type"] = infer_location_type(loc)
    if "conflict_type" not in tags:
        tags["conflict_type"] = infer_conflict_type(str(scenario.get("player_input", "")))
    if "persona_archetype" not in tags:
        tags["persona_archetype"] = infer_persona_archetype(str(scenario.get("persona", "")))
    return tags


def analyze_scenario_dependence(scenarios_by_id: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(scenarios_by_id.values())
    n = len(rows)
    by_source: Dict[str, int] = {}
    by_signature: Dict[str, int] = {}
    by_source_rows: Dict[str, List[Dict[str, Any]]] = {}

    for row in rows:
        sid = str(row.get("scenario_id", "")).strip()
        source_id = str(row.get("source_scenario_id", "")).strip() or sid
        by_source[source_id] = by_source.get(source_id, 0) + 1
        by_source_rows.setdefault(source_id, []).append(row)

        tags = get_scenario_tags(row)
        signature = str(tags.get("template_signature", "")).strip()
        if not signature:
            signature = "|".join(
                [
                    str(tags.get("conflict_type", "")).strip() or "na",
                    str(tags.get("location_type", "")).strip() or "na",
                    str(tags.get("behavior_state", "")).strip() or "na",
                ]
            )
        by_signature[signature] = by_signature.get(signature, 0) + 1

    def effective_n(counter: Dict[str, int]) -> float:
        denom = float(sum(v * v for v in counter.values()))
        numer = float(n * n)
        return (numer / denom) if denom > 0 else 0.0

    within_source_input_jaccard: List[float] = []
    for srows in by_source_rows.values():
        if len(srows) <= 1:
            continue
        base = str(srows[0].get("player_input", ""))
        sims = [jaccard_overlap(base, str(r.get("player_input", ""))) for r in srows[1:]]
        if sims:
            within_source_input_jaccard.append(statistics.fmean(sims))

    return {
        "scenario_count": n,
        "unique_source_count": len(by_source),
        "unique_template_signature_count": len(by_signature),
        "source_distribution": dict(sorted(by_source.items())),
        "template_signature_distribution": dict(sorted(by_signature.items())),
        "template_signature_ratio": (len(by_signature) / float(n)) if n > 0 else 0.0,
        "effective_sample_size_by_source": effective_n(by_source),
        "effective_sample_size_by_template_signature": effective_n(by_signature),
        "source_max_share": (max(by_source.values()) / float(n)) if n > 0 and by_source else 0.0,
        "template_max_share": (max(by_signature.values()) / float(n)) if n > 0 and by_signature else 0.0,
        "mean_within_source_player_input_jaccard": (
            statistics.fmean(within_source_input_jaccard) if within_source_input_jaccard else float("nan")
        ),
    }


def summarize_scores_by_slice(
    scored_by_arm: Dict[str, List[Dict[str, Any]]],
    scenarios_by_id: Dict[str, Dict[str, Any]],
    slice_keys: Sequence[str],
    seed: int,
    include_bertscore: bool,
) -> Dict[str, Any]:
    chosen_metrics = [*BASE_METRICS, *GAMEPLAY_METRICS, "overall_quality"]
    chosen_metrics.extend(["context_keyword_coverage", "persona_keyword_coverage"])
    if include_bertscore:
        chosen_metrics.append("bertscore_f1")

    output: Dict[str, Any] = {"slice_keys": list(slice_keys), "arms": {}}
    for arm_id, rows in scored_by_arm.items():
        arm_payload: Dict[str, Any] = {}
        for slice_idx, slice_key in enumerate(slice_keys):
            buckets: Dict[str, List[Dict[str, Any]]] = {}
            for row in rows:
                sid = str(row.get("scenario_id", ""))
                scenario = scenarios_by_id.get(sid, {})
                tags = get_scenario_tags(scenario)
                value = str(tags.get(slice_key, "")).strip()
                if not value:
                    continue
                buckets.setdefault(value, []).append(row)

            slice_payload: Dict[str, Any] = {}
            for bucket_idx, (bucket_value, bucket_rows) in enumerate(sorted(buckets.items())):
                metric_payload: Dict[str, Any] = {"sample_count": len(bucket_rows)}
                for metric_idx, metric_name in enumerate(chosen_metrics):
                    vals = [float(r.get(metric_name, float("nan"))) for r in bucket_rows if metric_name in r]
                    vals = [x for x in vals if not math.isnan(x)]
                    metric_payload[metric_name] = bootstrap_mean_ci(
                        vals,
                        seed=seed + 211 * (slice_idx + 1) + 41 * (bucket_idx + 1) + metric_idx,
                    )
                slice_payload[bucket_value] = metric_payload
            arm_payload[slice_key] = slice_payload
        output["arms"][arm_id] = arm_payload
    return output


def cohen_kappa_from_labels(labels_a: Sequence[int], labels_b: Sequence[int]) -> float:
    n = min(len(labels_a), len(labels_b))
    if n == 0:
        return float("nan")
    a = [int(x) for x in labels_a[:n]]
    b = [int(x) for x in labels_b[:n]]
    labels = sorted(set(a) | set(b))
    if not labels:
        return float("nan")

    observed = sum(1 for x, y in zip(a, b) if x == y) / float(n)
    pe = 0.0
    for label in labels:
        pa = sum(1 for x in a if x == label) / float(n)
        pb = sum(1 for y in b if y == label) / float(n)
        pe += pa * pb
    if pe >= 1.0:
        return float("nan")
    return (observed - pe) / (1.0 - pe)


def read_human_eval_rows(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    if path.suffix.lower() in (".csv", ".tsv"):
        delimiter = "," if path.suffix.lower() == ".csv" else "\t"
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            return [dict(row) for row in reader]
    raise ValueError(f"Unsupported human-eval format: {path}")


def normalize_human_score(value: Any, scale_max: float) -> float:
    try:
        raw = float(value)
    except Exception:
        return float("nan")
    if math.isnan(raw):
        return float("nan")
    if raw < 0.0:
        return float("nan")
    if raw <= 1.0:
        return max(0.0, min(1.0, raw))
    denom = max(1.0, float(scale_max))
    return max(0.0, min(1.0, raw / denom))


def analyze_human_eval(
    rows: List[Dict[str, Any]],
    metrics: Sequence[str],
    scale_max: float,
    target_arm: str,
    baseline_arms: Sequence[str],
    seed: int,
) -> Dict[str, Any]:
    normalized: List[Dict[str, Any]] = []
    for row in rows:
        scenario_id = str(row.get("scenario_id", "")).strip()
        arm_id = str(row.get("arm_id", "")).strip()
        annotator = str(row.get("annotator_id", row.get("annotator", row.get("user", "")))).strip()
        if not scenario_id or not arm_id or not annotator:
            continue
        item: Dict[str, Any] = {
            "scenario_id": scenario_id,
            "arm_id": arm_id,
            "annotator_id": annotator,
        }
        has_metric = False
        for metric in metrics:
            val = normalize_human_score(row.get(metric, float("nan")), scale_max=scale_max)
            if not math.isnan(val):
                item[metric] = val
                has_metric = True
        if has_metric:
            normalized.append(item)

    out: Dict[str, Any] = {
        "row_count": len(normalized),
        "metric_scale_max": scale_max,
        "metrics": list(metrics),
        "arms": {},
        "agreement": {},
        "paired_deltas": {},
        "preferences": {},
    }
    if not normalized:
        return out

    by_arm: Dict[str, List[Dict[str, Any]]] = {}
    for row in normalized:
        by_arm.setdefault(str(row.get("arm_id", "")), []).append(row)

    for arm_id, arm_rows in sorted(by_arm.items()):
        metric_summary: Dict[str, Any] = {"sample_count": len(arm_rows)}
        for metric_idx, metric in enumerate(metrics):
            vals = [float(r.get(metric, float("nan"))) for r in arm_rows if metric in r]
            vals = [x for x in vals if not math.isnan(x)]
            metric_summary[metric] = bootstrap_mean_ci(vals, seed=seed + 17 * (metric_idx + 1) + len(arm_id))
        out["arms"][arm_id] = metric_summary

    # Inter-rater agreement: pairwise kappa over shared (scenario, arm) items per metric.
    annotators = sorted({str(r.get("annotator_id", "")) for r in normalized})
    for metric_idx, metric in enumerate(metrics):
        ratings: Dict[Tuple[str, str], Dict[str, int]] = {}
        for row in normalized:
            if metric not in row:
                continue
            key = (str(row.get("scenario_id", "")), str(row.get("arm_id", "")))
            score_bin = int(round(float(row.get(metric, 0.0)) * 4.0))
            ratings.setdefault(key, {})[str(row.get("annotator_id", ""))] = score_bin

        pairwise: Dict[str, float] = {}
        pair_vals: List[float] = []
        for a1, a2 in combinations(annotators, 2):
            labs1: List[int] = []
            labs2: List[int] = []
            for key, mapping in ratings.items():
                if a1 in mapping and a2 in mapping:
                    labs1.append(int(mapping[a1]))
                    labs2.append(int(mapping[a2]))
            if not labs1:
                continue
            kappa = cohen_kappa_from_labels(labs1, labs2)
            pairwise[f"{a1}__{a2}"] = kappa
            if not math.isnan(kappa):
                pair_vals.append(kappa)

        out["agreement"][metric] = {
            "pairwise_kappa": pairwise,
            "mean_pairwise_kappa": (statistics.fmean(pair_vals) if pair_vals else float("nan")),
            "pair_count": len(pairwise),
        }

    # Paired deltas by scenario-level mean score across annotators.
    scenario_arm_mean: Dict[Tuple[str, str], Dict[str, float]] = {}
    for metric in metrics:
        grouped: Dict[Tuple[str, str], List[float]] = {}
        for row in normalized:
            if metric not in row:
                continue
            key = (str(row.get("scenario_id", "")), str(row.get("arm_id", "")))
            grouped.setdefault(key, []).append(float(row.get(metric, 0.0)))
        for key, values in grouped.items():
            payload = scenario_arm_mean.setdefault(key, {})
            payload[metric] = statistics.fmean(values)

    for baseline_idx, baseline_arm in enumerate(baseline_arms):
        comparison_key = f"{target_arm}_vs_{baseline_arm}"
        metric_deltas: Dict[str, Any] = {}
        for metric_idx, metric in enumerate(metrics):
            target_vals: List[float] = []
            baseline_vals: List[float] = []
            scenario_ids = sorted({sid for sid, _ in scenario_arm_mean.keys()})
            for sid in scenario_ids:
                target_key = (sid, target_arm)
                base_key = (sid, baseline_arm)
                if target_key not in scenario_arm_mean or base_key not in scenario_arm_mean:
                    continue
                if metric not in scenario_arm_mean[target_key] or metric not in scenario_arm_mean[base_key]:
                    continue
                target_vals.append(float(scenario_arm_mean[target_key][metric]))
                baseline_vals.append(float(scenario_arm_mean[base_key][metric]))
            if not target_vals:
                continue
            metric_deltas[metric] = paired_bootstrap_delta_ci(
                target_vals,
                baseline_vals,
                seed=seed + 601 * (baseline_idx + 1) + 29 * (metric_idx + 1),
            )
        out["paired_deltas"][comparison_key] = metric_deltas

    # Derived preferences: per (scenario, annotator), compare overall_quality-like metric.
    pref_metric = "overall_quality" if "overall_quality" in metrics else metrics[-1]
    score_lookup: Dict[Tuple[str, str, str], float] = {}
    for row in normalized:
        if pref_metric not in row:
            continue
        key = (
            str(row.get("scenario_id", "")),
            str(row.get("annotator_id", "")),
            str(row.get("arm_id", "")),
        )
        score_lookup[key] = float(row.get(pref_metric, 0.0))

    for baseline_idx, baseline_arm in enumerate(baseline_arms):
        comparison_key = f"{target_arm}_vs_{baseline_arm}"
        outcomes: List[float] = []
        wins = 0
        losses = 0
        ties = 0
        keys = sorted({(sid, ann) for sid, ann, _ in score_lookup.keys()})
        for sid, ann in keys:
            tk = (sid, ann, target_arm)
            bk = (sid, ann, baseline_arm)
            if tk not in score_lookup or bk not in score_lookup:
                continue
            tv = score_lookup[tk]
            bv = score_lookup[bk]
            if tv > bv:
                wins += 1
                outcomes.append(1.0)
            elif tv < bv:
                losses += 1
                outcomes.append(0.0)
            else:
                ties += 1
                outcomes.append(0.5)

        strict = [x for x in outcomes if x in (0.0, 1.0)]
        out["preferences"][comparison_key] = {
            "n": len(outcomes),
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "soft_win_rate": bootstrap_mean_ci(outcomes, seed=seed + 701 * (baseline_idx + 1)),
            "strict_non_tie_win_rate": bootstrap_mean_ci(strict, seed=seed + 709 * (baseline_idx + 1))
            if strict
            else {
                "n": 0,
                "mean": float("nan"),
                "median": float("nan"),
                "p95": float("nan"),
                "ci95_low": float("nan"),
                "ci95_high": float("nan"),
            },
        }

    return out


def render_human_eval_report(path: Path, payload: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# Human Evaluation Report")
    lines.append("")
    lines.append(f"- Total normalized rows: `{payload.get('row_count', 0)}`")
    lines.append(f"- Metrics: `{', '.join(payload.get('metrics', []))}`")
    lines.append("")

    lines.append("## Arm Summary")
    lines.append("| Arm | Metric | Mean | 95% CI |")
    lines.append("|---|---|---:|---:|")
    for arm_id, arm_data in payload.get("arms", {}).items():
        for metric in payload.get("metrics", []):
            m = arm_data.get(metric, {})
            lines.append(
                f"| {arm_id} | {metric} | {m.get('mean', float('nan')):.4f} | "
                f"({m.get('ci95_low', float('nan')):.4f}, {m.get('ci95_high', float('nan')):.4f}) |"
            )
    lines.append("")

    lines.append("## Agreement")
    lines.append("| Metric | Mean Pairwise Kappa | Pair Count |")
    lines.append("|---|---:|---:|")
    for metric, row in payload.get("agreement", {}).items():
        lines.append(
            f"| {metric} | {row.get('mean_pairwise_kappa', float('nan')):.4f} | {row.get('pair_count', 0)} |"
        )
    lines.append("")

    lines.append("## Preference Wins")
    lines.append("| Comparison | Soft Win Rate | Strict Non-tie Win Rate |")
    lines.append("|---|---:|---:|")
    for comp, row in payload.get("preferences", {}).items():
        soft = row.get("soft_win_rate", {})
        strict = row.get("strict_non_tie_win_rate", {})
        lines.append(
            f"| {comp} | {soft.get('mean', float('nan')):.4f} | {strict.get('mean', float('nan')):.4f} |"
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def analyze_errors(scored_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    categories = {
        "context_miss": 0,
        "persona_drift": 0,
        "low_naturalness": 0,
        "too_short": 0,
    }
    per_case: List[Dict[str, Any]] = []
    for row in scored_rows:
        response = str(row.get("response", ""))
        labels: List[str] = []
        if float(row.get("context_relevance", 0.0)) < 0.35:
            labels.append("context_miss")
        if float(row.get("persona_consistency", 0.0)) < 0.35:
            labels.append("persona_drift")
        if float(row.get("naturalness", 0.0)) < 0.40:
            labels.append("low_naturalness")
        if len(tokenize(response)) < 10:
            labels.append("too_short")

        for label in labels:
            categories[label] += 1
        per_case.append(
            {
                "scenario_id": row.get("scenario_id"),
                "labels": labels,
            }
        )

    return {
        "counts": categories,
        "per_case": per_case,
    }


def analyze_multi_turn_contradictions(
    scored_by_arm: Dict[str, List[Dict[str, Any]]],
    scenarios_by_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"arms": {}}
    for arm_id, rows in scored_by_arm.items():
        by_source: Dict[str, List[float]] = {}
        for row in rows:
            sid = str(row.get("scenario_id", "")).strip()
            scenario = scenarios_by_id.get(sid, {})
            source_id = str(scenario.get("source_scenario_id", "")).strip() or sid
            rate = float(row.get("multi_turn_contradiction_rate", 0.0) or 0.0)
            by_source.setdefault(source_id, []).append(rate)

        source_payload: Dict[str, Any] = {}
        source_rates: List[float] = []
        contradicted_sources = 0
        for source_id, vals in sorted(by_source.items()):
            if not vals:
                continue
            source_rate = float(statistics.fmean(vals))
            source_payload[source_id] = {
                "n": len(vals),
                "contradiction_rate": source_rate,
                "contradiction_safety": 1.0 - source_rate,
            }
            source_rates.append(source_rate)
            if source_rate > 0.0:
                contradicted_sources += 1

        overall_rate = float(statistics.fmean(source_rates)) if source_rates else float("nan")
        out["arms"][arm_id] = {
            "source_count": len(source_payload),
            "contradicted_source_count": contradicted_sources,
            "multi_turn_contradiction_rate": overall_rate,
            "multi_turn_contradiction_safety": (1.0 - overall_rate) if not math.isnan(overall_rate) else float("nan"),
            "by_source": source_payload,
        }
    return out


def summarize_operational_metrics(
    responses_by_arm: Dict[str, List[Dict[str, Any]]],
    scenarios_by_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    output: Dict[str, Any] = {"arms": {}}
    for arm_id, rows in responses_by_arm.items():
        total = len(rows)
        if total == 0:
            output["arms"][arm_id] = {
                "total_requests": 0,
                "timeout_rate": float("nan"),
                "error_rate": float("nan"),
                "fallback_rate": float("nan"),
                "first_pass_accept_rate": float("nan"),
                "rewrite_attempt_rate": float("nan"),
                "retry_rate": float("nan"),
                "multi_retry_rate": float("nan"),
                "rewrite_failure_rate": float("nan"),
                "by_behavior_state": {},
            }
            continue

        timeout_count = 0
        error_count = 0
        fallback_count = 0
        first_pass_count = 0
        rewrite_attempt_count = 0
        retry_count = 0
        multi_retry_count = 0
        rewrite_failure_count = 0
        by_behavior: Dict[str, Dict[str, int]] = {}
        source_counts: Dict[str, int] = {}

        for row in rows:
            ok = bool(row.get("ok"))
            err = str(row.get("error", "")).lower()
            if (not ok) and ("timeout" in err):
                timeout_count += 1
            if not ok:
                error_count += 1
            if bool(row.get("response_fallback")) or str(row.get("response_control_source", "")) == "fallback":
                fallback_count += 1

            source = str(row.get("response_control_source", "")).strip()
            if source:
                source_counts[source] = int(source_counts.get(source, 0)) + 1
            if source in ("raw", "raw_relaxed", "raw_adaptive", "raw_grounded_repair"):
                first_pass_count += 1

            attempts = int(row.get("rewrite_attempts", 0) or 0)
            success_attempts = int(row.get("rewrite_successful_attempts", 0) or 0)
            if attempts > 0:
                rewrite_attempt_count += 1
                retry_count += 1
                if attempts > 1:
                    multi_retry_count += 1
                if success_attempts <= 0:
                    rewrite_failure_count += 1

            sid = str(row.get("scenario_id", "")).strip()
            scenario = scenarios_by_id.get(sid, {})
            tags = get_scenario_tags(scenario)
            behavior = str(tags.get("behavior_state", "unknown")).strip() or "unknown"
            bucket = by_behavior.setdefault(
                behavior,
                {"count": 0, "fallback": 0, "timeout": 0, "errors": 0, "rewrite_attempted": 0},
            )
            bucket["count"] += 1
            if bool(row.get("response_fallback")) or source == "fallback":
                bucket["fallback"] += 1
            if (not ok) and ("timeout" in err):
                bucket["timeout"] += 1
            if not ok:
                bucket["errors"] += 1
            if attempts > 0:
                bucket["rewrite_attempted"] += 1

        by_behavior_rates: Dict[str, Any] = {}
        for behavior, counts in sorted(by_behavior.items()):
            denom = max(1, int(counts.get("count", 0)))
            by_behavior_rates[behavior] = {
                "count": denom,
                "fallback_rate": float(counts.get("fallback", 0)) / float(denom),
                "timeout_rate": float(counts.get("timeout", 0)) / float(denom),
                "error_rate": float(counts.get("errors", 0)) / float(denom),
                "rewrite_attempt_rate": float(counts.get("rewrite_attempted", 0)) / float(denom),
            }

        output["arms"][arm_id] = {
            "total_requests": total,
            "timeout_rate": timeout_count / float(total),
            "error_rate": error_count / float(total),
            "fallback_rate": fallback_count / float(total),
            "first_pass_accept_rate": first_pass_count / float(total),
            "rewrite_attempt_rate": rewrite_attempt_count / float(total),
            "retry_rate": retry_count / float(total),
            "multi_retry_rate": multi_retry_count / float(total),
            "rewrite_failure_rate": rewrite_failure_count / float(total),
            "control_source_breakdown": dict(sorted(source_counts.items(), key=lambda kv: kv[0])),
            "by_behavior_state": by_behavior_rates,
        }
    return output


def evaluate_operational_gate(
    operational_metrics: Dict[str, Any],
    arm_id: str,
    max_fallback_rate: float,
    max_retry_rate: float,
    min_first_pass_accept_rate: float,
) -> Dict[str, Any]:
    arms_payload = operational_metrics.get("arms", {})
    arm_payload = arms_payload.get(arm_id, {})

    fallback_rate = float(arm_payload.get("fallback_rate", float("nan")))
    retry_rate = float(arm_payload.get("retry_rate", float("nan")))
    first_pass_accept_rate = float(arm_payload.get("first_pass_accept_rate", float("nan")))

    checks = [
        {
            "name": "fallback_rate",
            "observed": fallback_rate,
            "rule": f"<= {max_fallback_rate:.4f}",
            "pass": (not math.isnan(fallback_rate)) and (fallback_rate <= max_fallback_rate),
        },
        {
            "name": "retry_rate",
            "observed": retry_rate,
            "rule": f"<= {max_retry_rate:.4f}",
            "pass": (not math.isnan(retry_rate)) and (retry_rate <= max_retry_rate),
        },
        {
            "name": "first_pass_accept_rate",
            "observed": first_pass_accept_rate,
            "rule": f">= {min_first_pass_accept_rate:.4f}",
            "pass": (not math.isnan(first_pass_accept_rate))
            and (first_pass_accept_rate >= min_first_pass_accept_rate),
        },
    ]
    passed = all(bool(item.get("pass")) for item in checks)
    return {
        "arm_id": arm_id,
        "pass": passed,
        "thresholds": {
            "max_fallback_rate": max_fallback_rate,
            "max_retry_rate": max_retry_rate,
            "min_first_pass_accept_rate": min_first_pass_accept_rate,
        },
        "checks": checks,
    }


def render_report(
    output_path: Path,
    run_id: str,
    scenarios_path: Path,
    arms: List[EvaluationArm],
    summary: Dict[str, Any],
    deltas: Dict[str, Any],
    paired_deltas: Dict[str, Any],
    bertscore_meta: Dict[str, Any],
    win_rates: Optional[Dict[str, Any]] = None,
    slice_summary: Optional[Dict[str, Any]] = None,
    human_eval: Optional[Dict[str, Any]] = None,
    operational_metrics: Optional[Dict[str, Any]] = None,
    scenario_dependence: Optional[Dict[str, Any]] = None,
    multi_turn_contradictions: Optional[Dict[str, Any]] = None,
) -> None:
    lines: List[str] = []
    lines.append("# Proposal Alignment Evaluation Report")
    lines.append("")
    lines.append(f"- Run ID: `{run_id}`")
    lines.append(f"- Generated: `{utc_iso()}`")
    lines.append(f"- Scenarios: `{scenarios_path}`")
    lines.append(f"- Scenario count: `{len(read_jsonl(scenarios_path))}`")
    lines.append("")
    lines.append("## Evaluation Arms")
    for arm in arms:
        lines.append(
            f"- `{arm.arm_id}`: model `{arm.model}`, dynamic_context={'on' if arm.include_dynamic_context else 'off'}, "
            f"response_control={'on' if arm.use_response_control else 'off'}"
        )
    lines.append("")
    lines.append("## Metric Summary (mean, 95% CI)")
    lines.append("| Arm | Context Relevance | Persona Consistency | Naturalness | Overall | BERTScore F1 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for arm in arms:
        arm_summary = summary.get(arm.arm_id, {})
        c = arm_summary.get("context_relevance", {})
        p = arm_summary.get("persona_consistency", {})
        n = arm_summary.get("naturalness", {})
        o = arm_summary.get("overall_quality", {})
        b = arm_summary.get("bertscore_f1", {})
        if b:
            b_text = f"{b.get('mean', float('nan')):.4f}"
        else:
            b_text = "n/a"
        lines.append(
            "| "
            + arm.arm_id
            + " | "
            + f"{c.get('mean', float('nan')):.4f} ({c.get('ci95_low', float('nan')):.4f}, {c.get('ci95_high', float('nan')):.4f})"
            + " | "
            + f"{p.get('mean', float('nan')):.4f} ({p.get('ci95_low', float('nan')):.4f}, {p.get('ci95_high', float('nan')):.4f})"
            + " | "
            + f"{n.get('mean', float('nan')):.4f} ({n.get('ci95_low', float('nan')):.4f}, {n.get('ci95_high', float('nan')):.4f})"
            + " | "
            + f"{o.get('mean', float('nan')):.4f} ({o.get('ci95_low', float('nan')):.4f}, {o.get('ci95_high', float('nan')):.4f})"
            + " | "
            + b_text
            + " |"
        )
    lines.append("")

    lines.append("## Game-facing Outcome Metrics (mean, 95% CI)")
    lines.append(
        "| Arm | Quest-state Correctness | Lore Consistency | Contradiction Safety | Objective Completion Support | Gameplay Usefulness | Time-pressure Acceptability |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for arm in arms:
        arm_summary = summary.get(arm.arm_id, {})
        q = arm_summary.get("quest_state_correctness", {})
        l = arm_summary.get("lore_consistency", {})
        c = arm_summary.get("multi_turn_contradiction_safety", {})
        ocs = arm_summary.get("objective_completion_support", {})
        g = arm_summary.get("gameplay_usefulness", {})
        t = arm_summary.get("time_pressure_acceptability", {})
        lines.append(
            f"| {arm.arm_id} | "
            f"{q.get('mean', float('nan')):.4f} ({q.get('ci95_low', float('nan')):.4f}, {q.get('ci95_high', float('nan')):.4f}) | "
            f"{l.get('mean', float('nan')):.4f} ({l.get('ci95_low', float('nan')):.4f}, {l.get('ci95_high', float('nan')):.4f}) | "
            f"{c.get('mean', float('nan')):.4f} ({c.get('ci95_low', float('nan')):.4f}, {c.get('ci95_high', float('nan')):.4f}) | "
            f"{ocs.get('mean', float('nan')):.4f} ({ocs.get('ci95_low', float('nan')):.4f}, {ocs.get('ci95_high', float('nan')):.4f}) | "
            f"{g.get('mean', float('nan')):.4f} ({g.get('ci95_low', float('nan')):.4f}, {g.get('ci95_high', float('nan')):.4f}) | "
            f"{t.get('mean', float('nan')):.4f} ({t.get('ci95_low', float('nan')):.4f}, {t.get('ci95_high', float('nan')):.4f}) |"
        )
    lines.append("")
    lines.append("- Multi-turn contradiction rate is reported as `1 - contradiction_safety` in row-level outputs.")
    lines.append("")

    lines.append("## Deltas vs Baselines")
    lines.append("| Comparison | Metric | Absolute Delta | Relative Delta |")
    lines.append("|---|---|---:|---:|")
    for comp_name, comp_metrics in deltas.items():
        for metric_name, vals in comp_metrics.items():
            lines.append(
                f"| {comp_name} | {metric_name} | {vals.get('absolute_delta', float('nan')):.4f} | {vals.get('relative_delta', float('nan')):.4f} |"
            )
    lines.append("")

    if paired_deltas:
        lines.append("## Paired Bootstrap Delta Significance")
        lines.append(
            "| Comparison | Metric | Mean Delta | 95% CI | p(delta<=0) | Cluster Mean Delta | Cluster 95% CI | Cluster p(delta<=0) |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for comp_name, comp_metrics in paired_deltas.items():
            for metric_name, vals in comp_metrics.items():
                cluster = vals.get("cluster_bootstrap", {}) if isinstance(vals, dict) else {}
                if cluster:
                    cluster_mean = f"{cluster.get('mean_delta', float('nan')):.4f}"
                    cluster_ci = f"({cluster.get('ci95_low', float('nan')):.4f}, {cluster.get('ci95_high', float('nan')):.4f})"
                    cluster_p = f"{cluster.get('p_delta_le_0', float('nan')):.4f}"
                else:
                    cluster_mean = "n/a"
                    cluster_ci = "n/a"
                    cluster_p = "n/a"
                lines.append(
                    f"| {comp_name} | {metric_name} | {vals.get('mean_delta', float('nan')):.4f} | "
                    f"({vals.get('ci95_low', float('nan')):.4f}, {vals.get('ci95_high', float('nan')):.4f}) | "
                    f"{vals.get('p_delta_le_0', float('nan')):.4f} | "
                    f"{cluster_mean} | {cluster_ci} | {cluster_p} |"
                )
        lines.append("")

    if win_rates:
        lines.append("## Pairwise Win Rates")
        lines.append("| Comparison | Metric | Wins | Losses | Ties | Soft Win Rate | Strict Non-tie Win Rate |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for comp_name, comp_metrics in win_rates.items():
            for metric_name, vals in comp_metrics.items():
                soft = vals.get("soft_win_rate", {})
                strict = vals.get("strict_non_tie_win_rate", {})
                lines.append(
                    f"| {comp_name} | {metric_name} | {vals.get('wins', 0)} | {vals.get('losses', 0)} | "
                    f"{vals.get('ties', 0)} | {soft.get('mean', float('nan')):.4f} | {strict.get('mean', float('nan')):.4f} |"
                )
        lines.append("")

    if slice_summary:
        lines.append("## Scenario Slice Coverage")
        lines.append(f"- Slice keys: `{', '.join(slice_summary.get('slice_keys', []))}`")
        lines.append("- Detailed slice metrics are published in `slice_summary.json`.")
        lines.append("")

    if operational_metrics:
        lines.append("## Operational Metrics")
        lines.append("| Arm | Timeout Rate | Error Rate | Fallback Rate | First-pass Accept Rate | Retry Rate |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for arm in arms:
            payload = operational_metrics.get("arms", {}).get(arm.arm_id, {})
            lines.append(
                f"| {arm.arm_id} | {payload.get('timeout_rate', float('nan')):.4f} | "
                f"{payload.get('error_rate', float('nan')):.4f} | "
                f"{payload.get('fallback_rate', float('nan')):.4f} | "
                f"{payload.get('first_pass_accept_rate', float('nan')):.4f} | "
                f"{payload.get('retry_rate', float('nan')):.4f} |"
            )
        lines.append("")

    if scenario_dependence:
        lines.append("## Scenario Dependence Diagnostics")
        lines.append(f"- Unique source scenarios: `{scenario_dependence.get('unique_source_count', 0)}`")
        lines.append(
            f"- Unique template signatures: `{scenario_dependence.get('unique_template_signature_count', 0)}`"
        )
        lines.append(
            f"- Template signature ratio: `{scenario_dependence.get('template_signature_ratio', float('nan')):.4f}`"
        )
        lines.append(
            f"- Effective sample size by source clustering: "
            f"`{scenario_dependence.get('effective_sample_size_by_source', float('nan')):.2f}`"
        )
        lines.append(
            f"- Effective sample size by template-signature clustering: "
            f"`{scenario_dependence.get('effective_sample_size_by_template_signature', float('nan')):.2f}`"
        )
        lines.append("- Detailed diagnostics are published in `scenario_dependence.json`.")
        lines.append("")

    if multi_turn_contradictions:
        lines.append("## Multi-turn Contradiction")
        lines.append("| Arm | Contradiction Rate | Contradiction Safety | Contradicted Sources | Source Count |")
        lines.append("|---|---:|---:|---:|---:|")
        for arm in arms:
            payload = multi_turn_contradictions.get("arms", {}).get(arm.arm_id, {})
            lines.append(
                f"| {arm.arm_id} | {payload.get('multi_turn_contradiction_rate', float('nan')):.4f} | "
                f"{payload.get('multi_turn_contradiction_safety', float('nan')):.4f} | "
                f"{payload.get('contradicted_source_count', 0)} | {payload.get('source_count', 0)} |"
            )
        lines.append("- Detailed source-level values are published in `multi_turn_contradictions.json`.")
        lines.append("")

    if human_eval:
        lines.append("## Human Evaluation")
        lines.append(f"- Normalized rows: `{human_eval.get('row_count', 0)}`")
        lines.append("- Agreement and preference analysis is published in `human_eval_summary.json`.")
        lines.append("- A readable version is published in `human_eval_report.md`.")
        lines.append("")

    if bertscore_meta.get("available"):
        lines.append("- BERTScore status: enabled.")
    else:
        lines.append(f"- BERTScore status: unavailable ({bertscore_meta.get('reason', 'unknown')}).")
    lines.append("")
    lines.append(
        "This report covers proposal RO5 metrics plus game-facing outcomes: quest-state correctness, lore consistency, contradiction safety, objective completion support, gameplay usefulness, and time-pressure acceptability."
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run proposal-aligned evaluation and export artifacts.")
    parser.add_argument("--host", default="http://127.0.0.1:11434", help="Ollama host URL")
    parser.add_argument("--candidate-model", default="elara-npc:latest", help="Candidate fine-tuned model")
    parser.add_argument("--baseline-model", default="phi3:mini", help="Baseline generic model")
    parser.add_argument(
        "--baseline-models",
        default="",
        help="Optional comma-separated additional external baseline models (all no-context).",
    )
    parser.add_argument(
        "--skip-external-baselines",
        action="store_true",
        help="Skip external no-context baseline arms (useful for rapid control tuning).",
    )
    parser.add_argument("--scenarios", default="data/proposal_eval_scenarios_large.jsonl", help="Scenario JSONL path")
    parser.add_argument("--temperature", type=float, default=0.2, help="Generation temperature")
    parser.add_argument("--max-tokens", type=int, default=96, help="Max generated tokens")
    parser.add_argument("--repeats", type=int, default=1, help="Runs per scenario per arm")
    parser.add_argument(
        "--max-scenarios",
        type=int,
        default=0,
        help="Optional cap on scenario count (0 means use all scenarios). Uses deterministic shuffled sampling.",
    )
    parser.add_argument(
        "--min-template-signature-ratio",
        type=float,
        default=0.0,
        help="Optional lower bound for scenario template-signature diversity ratio.",
    )
    parser.add_argument(
        "--min-effective-source-n",
        type=float,
        default=0.0,
        help="Optional lower bound for effective sample size under source clustering.",
    )
    parser.add_argument(
        "--enforce-scenario-diversity",
        action="store_true",
        help="Fail fast when scenario dependence diagnostics violate configured thresholds.",
    )
    parser.add_argument("--seed", type=int, default=19, help="Random seed")
    parser.add_argument("--timeout-s", type=int, default=180, help="Generation timeout seconds")
    parser.add_argument(
        "--min-arm-success-rate",
        type=float,
        default=0.90,
        help="Minimum fraction of successful API calls required per arm.",
    )
    parser.add_argument(
        "--preflight-operational-gate",
        action="store_true",
        help="Enable hard operational gate on fallback/retry/first-pass acceptance before scoring.",
    )
    parser.add_argument(
        "--preflight-gate-arm",
        default="proposed_contextual_controlled",
        help="Arm id to validate for operational preflight gate.",
    )
    parser.add_argument(
        "--preflight-max-fallback-rate",
        type=float,
        default=0.45,
        help="Maximum allowed fallback rate for operational preflight gate.",
    )
    parser.add_argument(
        "--preflight-max-retry-rate",
        type=float,
        default=0.75,
        help="Maximum allowed retry rate for operational preflight gate.",
    )
    parser.add_argument(
        "--preflight-min-first-pass-accept-rate",
        type=float,
        default=0.25,
        help="Minimum required first-pass acceptance rate for operational preflight gate.",
    )
    parser.add_argument("--bertscore-lang", default="en", help="Language for BERTScore")
    parser.add_argument(
        "--bertscore-model-type",
        default="",
        help="Optional HF model id for BERTScore (default: bert-score auto model for language).",
    )
    parser.add_argument(
        "--bertscore-batch-size",
        type=int,
        default=16,
        help="BERTScore batch size.",
    )
    parser.add_argument(
        "--bertscore-cache-dir",
        default="",
        help="Optional local cache dir for deterministic BERTScore model downloads.",
    )
    parser.add_argument(
        "--disable-bertscore",
        action="store_true",
        help="Disable BERTScore computation for faster tuning runs.",
    )
    parser.add_argument(
        "--control-min-context-coverage",
        type=float,
        default=0.33,
        help="Minimum context keyword coverage for response control.",
    )
    parser.add_argument(
        "--control-min-persona-coverage",
        type=float,
        default=0.18,
        help="Minimum persona keyword coverage for response control.",
    )
    parser.add_argument(
        "--control-min-response-tokens",
        type=int,
        default=8,
        help="Minimum token length before a response is considered too short.",
    )
    parser.add_argument(
        "--control-rewrite-max-tokens",
        type=int,
        default=96,
        help="Max tokens for response-control rewrite pass.",
    )
    parser.add_argument(
        "--control-rewrite-candidates",
        type=int,
        default=3,
        help="Number of rewrite candidates to sample for response control.",
    )
    parser.add_argument(
        "--control-rewrite-temperature-step",
        type=float,
        default=0.15,
        help="Temperature step used to diversify rewrite candidates.",
    )
    parser.add_argument(
        "--control-rewrite-temperature",
        type=float,
        default=0.2,
        help="Temperature for response-control rewrite pass.",
    )
    parser.add_argument(
        "--control-rewrite-budget-ms",
        type=float,
        default=0.0,
        help="Max cumulative rewrite generation latency per turn in milliseconds (0 disables cap).",
    )
    parser.add_argument(
        "--control-rewrite-budget-multiplier",
        type=float,
        default=0.0,
        help=(
            "Optional per-turn rewrite latency cap as a multiplier of raw generation latency "
            "(e.g., 0.75 means rewrite budget <= 75%% of raw generation time)."
        ),
    )
    parser.add_argument(
        "--control-relaxed-context-coverage",
        type=float,
        default=0.18,
        help="Relaxed context floor for accepting best-effort outputs.",
    )
    parser.add_argument(
        "--control-relaxed-persona-coverage",
        type=float,
        default=0.09,
        help="Relaxed persona floor for accepting best-effort outputs.",
    )
    parser.add_argument(
        "--control-relaxed-candidate-score",
        type=float,
        default=0.44,
        help="Relaxed candidate score threshold before forcing fallback.",
    )
    parser.add_argument(
        "--control-min-rewrite-gain",
        type=float,
        default=0.015,
        help="Minimum score margin required to select best-effort rewrite over fallback.",
    )
    parser.add_argument(
        "--control-early-stop-score",
        type=float,
        default=0.70,
        help="Stop rewrite sampling early once this strict-pass score is achieved.",
    )
    parser.add_argument(
        "--control-adaptive-candidate-score",
        type=float,
        default=0.38,
        help="Adaptive first-pass acceptance score threshold before rewrite/fallback.",
    )
    parser.add_argument(
        "--control-adaptive-context-coverage",
        type=float,
        default=0.14,
        help="Adaptive context floor for first-pass acceptance.",
    )
    parser.add_argument(
        "--control-adaptive-persona-coverage",
        type=float,
        default=0.10,
        help="Adaptive persona floor for first-pass acceptance.",
    )
    parser.add_argument(
        "--control-adaptive-high-score",
        type=float,
        default=0.53,
        help="Raw-score threshold for high-confidence rewrite budget.",
    )
    parser.add_argument(
        "--control-adaptive-mid-score",
        type=float,
        default=0.40,
        help="Raw-score threshold for mid-confidence rewrite budget.",
    )
    parser.add_argument(
        "--control-adaptive-high-rewrites",
        type=int,
        default=1,
        help="Rewrite budget when raw score is in high-confidence regime.",
    )
    parser.add_argument(
        "--control-adaptive-mid-rewrites",
        type=int,
        default=2,
        help="Rewrite budget when raw score is in mid-confidence regime.",
    )
    parser.add_argument(
        "--control-adaptive-low-rewrites",
        type=int,
        default=3,
        help="Rewrite budget when raw score is below the mid-confidence regime.",
    )
    parser.add_argument(
        "--disable-control-low-confidence-retry-gain",
        action="store_true",
        help="Disable low-confidence rewrite early-abort gate based on first-attempt gain.",
    )
    parser.add_argument(
        "--control-low-confidence-retry-min-score-gain",
        type=float,
        default=0.01,
        help="Minimum first-attempt score gain required to continue low-confidence retries.",
    )
    parser.add_argument(
        "--control-low-confidence-retry-min-coverage-gain",
        type=float,
        default=0.02,
        help="Minimum first-attempt context/persona coverage gain required to continue low-confidence retries.",
    )
    parser.add_argument(
        "--disable-control-relaxed-acceptance",
        action="store_true",
        help="Disable relaxed acceptance and force strict thresholds before fallback.",
    )
    parser.add_argument(
        "--disable-control-adaptive-acceptance",
        action="store_true",
        help="Disable adaptive first-pass acceptance prior to rewrite/fallback.",
    )
    parser.add_argument(
        "--disable-control-behavior-adaptation",
        action="store_true",
        help="Disable behavior-state-specific control threshold adaptation.",
    )
    parser.add_argument(
        "--enable-control-intent-risk-adaptation",
        action="store_true",
        help="Enable intent-risk-aware control threshold adaptation.",
    )
    parser.add_argument(
        "--enable-control-latency-adaptation",
        action="store_true",
        help="Enable latency-pressure-aware control threshold adaptation.",
    )
    parser.add_argument(
        "--control-latency-relax-start-pressure",
        type=float,
        default=0.55,
        help="Latency pressure (raw_latency/timeout_budget) above which thresholds start relaxing.",
    )
    parser.add_argument(
        "--control-latency-relax-max-delta",
        type=float,
        default=0.12,
        help="Maximum threshold relaxation applied under severe latency pressure.",
    )
    parser.add_argument(
        "--enable-control-intent-focused-context",
        action="store_true",
        help="Enable intent-focused context keyword selection inside response control.",
    )
    parser.add_argument(
        "--control-intent-focus-min-keep",
        type=int,
        default=3,
        help="Minimum number of context keywords kept after intent-focused filtering.",
    )
    parser.add_argument(
        "--control-intent-focus-keep-ratio-low",
        type=float,
        default=0.45,
        help="Context keyword keep ratio for low-risk intents.",
    )
    parser.add_argument(
        "--control-intent-focus-keep-ratio-medium",
        type=float,
        default=0.65,
        help="Context keyword keep ratio for medium-risk intents.",
    )
    parser.add_argument(
        "--control-intent-focus-keep-ratio-high",
        type=float,
        default=1.0,
        help="Context keyword keep ratio for high-risk intents.",
    )
    parser.add_argument(
        "--control-intent-focus-min-relevance",
        type=float,
        default=0.20,
        help="Minimum top relevance score required before applying context-key filtering.",
    )
    parser.add_argument(
        "--enable-control-near-pass",
        action="store_true",
        help="Enable near-pass acceptance to reduce unnecessary rewrite/fallback when candidates are close to thresholds.",
    )
    parser.add_argument(
        "--control-near-pass-max-context-gap",
        type=float,
        default=0.05,
        help="Maximum context-coverage shortfall allowed for near-pass acceptance.",
    )
    parser.add_argument(
        "--control-near-pass-max-persona-gap",
        type=float,
        default=0.04,
        help="Maximum persona-coverage shortfall allowed for near-pass acceptance.",
    )
    parser.add_argument(
        "--control-near-pass-score-floor",
        type=float,
        default=0.34,
        help="Minimum candidate score required for near-pass acceptance.",
    )
    parser.add_argument(
        "--disable-control-near-pass-block-high-risk",
        action="store_true",
        help="Allow near-pass acceptance for high-risk intents (disabled by default for safety).",
    )
    parser.add_argument(
        "--disable-control-early-stop",
        action="store_true",
        help="Disable early stopping in rewrite sampling.",
    )
    parser.add_argument(
        "--disable-control-rewrite",
        action="store_true",
        help="Disable rewrite pass in response control and use fallback directly.",
    )
    parser.add_argument(
        "--disable-control-best-effort-rewrite",
        action="store_true",
        help="Disable best-effort rewrite acceptance when strict thresholds are missed.",
    )
    parser.add_argument(
        "--control-alt-profile",
        choices=[
            "none",
            "runtime_optimized",
            "quality_optimized",
            "risk_latency_aware",
            "hybrid_balanced",
            "intent_focus_adaptive",
            "blend_balanced",
            "custom",
        ],
        default="none",
        help=(
            "Optional second controlled architecture profile added as an extra arm for in-run "
            "architecture-vs-architecture comparisons."
        ),
    )
    parser.add_argument(
        "--control-alt-arm-id",
        default="proposed_contextual_controlled_alt",
        help="Arm ID for the optional alternate controlled profile.",
    )
    parser.add_argument(
        "--control-alt-overrides-file",
        default="",
        help="Optional JSON file with ControlConfig field overrides for the alternate profile.",
    )
    parser.add_argument(
        "--control-alt-overrides-json",
        default="",
        help="Optional JSON object string with ControlConfig overrides (applied after profile).",
    )
    parser.add_argument(
        "--target-arm",
        default="proposed_contextual_controlled",
        help="Arm used as target for external-baseline win-rate and human-eval comparisons.",
    )
    parser.add_argument(
        "--slice-keys",
        default="persona_archetype,conflict_type,location_type,behavior_state",
        help="Comma-separated scenario slice keys for per-slice reporting.",
    )
    parser.add_argument(
        "--human-eval-file",
        default="",
        help="Optional CSV/TSV/JSONL with human ratings (scenario_id, arm_id, annotator_id, metric columns).",
    )
    parser.add_argument(
        "--human-eval-metrics",
        default="context_relevance,persona_consistency,naturalness,overall_quality",
        help="Comma-separated metric columns to read from human-eval file.",
    )
    parser.add_argument(
        "--human-eval-scale-max",
        type=float,
        default=5.0,
        help="Maximum rating scale used in human-eval file when values are >1.",
    )
    parser.add_argument("--output-root", default="artifacts/proposal", help="Output root folder")
    args = parser.parse_args()

    scenarios_path = Path(args.scenarios)
    if not scenarios_path.exists():
        raise FileNotFoundError(f"Scenario file not found: {scenarios_path}")

    run_id = utc_stamp()
    run_dir = Path(args.output_root) / run_id
    metadata_dir = run_dir / "metadata"
    responses_dir = run_dir / "responses"
    scores_dir = run_dir / "scores"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    responses_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)

    scenarios = read_jsonl(scenarios_path)
    if int(args.max_scenarios) > 0 and int(args.max_scenarios) < len(scenarios):
        sampled = list(scenarios)
        rng = random.Random(args.seed + 17)
        rng.shuffle(sampled)
        scenarios = sampled[: int(args.max_scenarios)]
    scenarios_by_id = {str(row.get("scenario_id")): row for row in scenarios}
    scenario_dependence = analyze_scenario_dependence(scenarios_by_id)
    min_sig_ratio = max(0.0, float(args.min_template_signature_ratio))
    min_eff_source_n = max(0.0, float(args.min_effective_source_n))
    sig_ratio = float(scenario_dependence.get("template_signature_ratio", 0.0) or 0.0)
    eff_source_n = float(scenario_dependence.get("effective_sample_size_by_source", float("nan")))
    if bool(args.enforce_scenario_diversity):
        fail_reasons: List[str] = []
        if sig_ratio < min_sig_ratio:
            fail_reasons.append(
                f"template_signature_ratio={sig_ratio:.4f} < min_template_signature_ratio={min_sig_ratio:.4f}"
            )
        if (not math.isnan(eff_source_n)) and eff_source_n < min_eff_source_n:
            fail_reasons.append(
                f"effective_sample_size_by_source={eff_source_n:.2f} < min_effective_source_n={min_eff_source_n:.2f}"
            )
        if fail_reasons:
            raise RuntimeError("Scenario diversity gate failed: " + "; ".join(fail_reasons))
    extra_baselines = parse_list_arg(str(args.baseline_models))
    baseline_models: List[str] = []
    if not bool(args.skip_external_baselines):
        for model_name in [str(args.baseline_model)] + extra_baselines:
            if model_name.lower() == str(args.candidate_model).lower():
                continue
            if model_name.lower() in {m.lower() for m in baseline_models}:
                continue
            baseline_models.append(model_name)
        if not baseline_models:
            baseline_models = [str(args.baseline_model)]

    baseline_arm_ids: List[str] = []
    for idx, model_name in enumerate(baseline_models):
        if idx == 0:
            baseline_arm_ids.append("baseline_no_context")
        else:
            baseline_arm_ids.append(f"baseline_no_context_{sanitize_model_id(model_name)}")

    base_control_config = ControlConfig(
        min_context_coverage=float(args.control_min_context_coverage),
        min_persona_coverage=float(args.control_min_persona_coverage),
        min_response_tokens=max(1, int(args.control_min_response_tokens)),
        rewrite_temperature=float(args.control_rewrite_temperature),
        rewrite_max_tokens=int(args.control_rewrite_max_tokens),
        rewrite_candidates=max(1, int(args.control_rewrite_candidates)),
        rewrite_temperature_step=float(args.control_rewrite_temperature_step),
        early_stop_on_pass=not bool(args.disable_control_early_stop),
        early_stop_score=float(args.control_early_stop_score),
        allow_relaxed_acceptance=not bool(args.disable_control_relaxed_acceptance),
        relaxed_context_coverage=max(0.0, float(args.control_relaxed_context_coverage)),
        relaxed_persona_coverage=max(0.0, float(args.control_relaxed_persona_coverage)),
        relaxed_candidate_score=max(0.0, min(1.0, float(args.control_relaxed_candidate_score))),
        min_rewrite_gain=max(0.0, float(args.control_min_rewrite_gain)),
        enable_rewrite=not bool(args.disable_control_rewrite),
        allow_best_effort_rewrite=not bool(args.disable_control_best_effort_rewrite),
        behavior_adaptation_enabled=not bool(args.disable_control_behavior_adaptation),
        adaptive_acceptance_enabled=not bool(args.disable_control_adaptive_acceptance),
        adaptive_candidate_score=max(0.0, min(1.0, float(args.control_adaptive_candidate_score))),
        adaptive_context_coverage=max(0.0, min(1.0, float(args.control_adaptive_context_coverage))),
        adaptive_persona_coverage=max(0.0, min(1.0, float(args.control_adaptive_persona_coverage))),
        adaptive_high_confidence_score=max(0.0, min(1.0, float(args.control_adaptive_high_score))),
        adaptive_mid_confidence_score=max(0.0, min(1.0, float(args.control_adaptive_mid_score))),
        adaptive_high_confidence_rewrites=max(1, int(args.control_adaptive_high_rewrites)),
        adaptive_mid_confidence_rewrites=max(1, int(args.control_adaptive_mid_rewrites)),
        adaptive_low_confidence_rewrites=max(1, int(args.control_adaptive_low_rewrites)),
        low_confidence_retry_requires_gain=not bool(args.disable_control_low_confidence_retry_gain),
        low_confidence_retry_min_score_gain=max(0.0, float(args.control_low_confidence_retry_min_score_gain)),
        low_confidence_retry_min_coverage_gain=max(
            0.0, float(args.control_low_confidence_retry_min_coverage_gain)
        ),
        intent_risk_adaptation_enabled=bool(args.enable_control_intent_risk_adaptation),
        latency_adaptation_enabled=bool(args.enable_control_latency_adaptation),
        latency_relax_start_pressure=max(0.0, min(1.0, float(args.control_latency_relax_start_pressure))),
        latency_relax_max_delta=max(0.0, min(0.25, float(args.control_latency_relax_max_delta))),
        intent_focused_context_enabled=bool(args.enable_control_intent_focused_context),
        intent_focus_min_keep=max(1, int(args.control_intent_focus_min_keep)),
        intent_focus_keep_ratio_low=max(0.2, min(1.0, float(args.control_intent_focus_keep_ratio_low))),
        intent_focus_keep_ratio_medium=max(0.2, min(1.0, float(args.control_intent_focus_keep_ratio_medium))),
        intent_focus_keep_ratio_high=max(0.2, min(1.0, float(args.control_intent_focus_keep_ratio_high))),
        intent_focus_min_relevance=max(0.0, min(1.0, float(args.control_intent_focus_min_relevance))),
        near_pass_enabled=bool(args.enable_control_near_pass),
        near_pass_max_context_gap=max(0.0, min(0.5, float(args.control_near_pass_max_context_gap))),
        near_pass_max_persona_gap=max(0.0, min(0.5, float(args.control_near_pass_max_persona_gap))),
        near_pass_score_floor=max(0.0, min(1.0, float(args.control_near_pass_score_floor))),
        near_pass_block_high_risk=not bool(args.disable_control_near_pass_block_high_risk),
    )
    base_control_config = normalize_control_config(base_control_config)
    control_rewrite_budget_ms = max(0.0, float(args.control_rewrite_budget_ms))
    control_rewrite_budget_multiplier = max(0.0, float(args.control_rewrite_budget_multiplier))
    alt_profile = str(args.control_alt_profile).strip().lower()
    alt_arm_id = str(args.control_alt_arm_id).strip() or "proposed_contextual_controlled_alt"
    alt_overrides: Dict[str, Any] = {}
    alt_overrides_file = str(args.control_alt_overrides_file).strip()
    if alt_overrides_file:
        override_path = Path(alt_overrides_file)
        if not override_path.exists():
            raise FileNotFoundError(f"control-alt-overrides-file not found: {override_path}")
        loaded = json.loads(override_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("control-alt-overrides-file must contain a JSON object.")
        alt_overrides.update(loaded)
    alt_overrides_json = str(args.control_alt_overrides_json).strip()
    if alt_overrides_json:
        loaded_inline = json.loads(alt_overrides_json)
        if not isinstance(loaded_inline, dict):
            raise ValueError("control-alt-overrides-json must be a JSON object.")
        alt_overrides.update(loaded_inline)
    if alt_arm_id in {"proposed_contextual", "candidate_no_context", "proposed_contextual_controlled"}:
        raise ValueError(f"control-alt-arm-id collides with reserved arm id: {alt_arm_id}")
    alt_control_config: Optional[ControlConfig] = None
    if alt_profile != "none":
        alt_control_config = build_alternate_control_config(base_control_config, alt_profile)
        alt_control_config = apply_control_overrides(alt_control_config, alt_overrides)

    arms = [
        EvaluationArm(
            arm_id="proposed_contextual_controlled",
            model=args.candidate_model,
            include_dynamic_context=True,
            use_response_control=True,
            control_profile="default",
        ),
        EvaluationArm(
            arm_id="proposed_contextual",
            model=args.candidate_model,
            include_dynamic_context=True,
        ),
        EvaluationArm(
            arm_id="candidate_no_context",
            model=args.candidate_model,
            include_dynamic_context=False,
        ),
    ]
    if alt_control_config is not None:
        arms.insert(
            1,
            EvaluationArm(
                arm_id=alt_arm_id,
                model=args.candidate_model,
                include_dynamic_context=True,
                use_response_control=True,
                control_profile=alt_profile,
            ),
        )
    for arm_id, model_name in zip(baseline_arm_ids, baseline_models):
        arms.append(
            EvaluationArm(
                arm_id=arm_id,
                model=model_name,
                include_dynamic_context=False,
            )
        )
    arm_ids_seen: Dict[str, int] = {}
    for arm in arms:
        arm_ids_seen[arm.arm_id] = int(arm_ids_seen.get(arm.arm_id, 0)) + 1
    duplicate_arm_ids = sorted([k for k, v in arm_ids_seen.items() if v > 1])
    if duplicate_arm_ids:
        raise ValueError(f"Duplicate arm IDs detected: {duplicate_arm_ids}")

    response_control_config_payload = control_config_payload(
        config=base_control_config,
        rewrite_budget_ms=control_rewrite_budget_ms,
        rewrite_budget_multiplier=control_rewrite_budget_multiplier,
    )
    control_profiles_payload: Dict[str, Dict[str, Any]] = {"default": response_control_config_payload}
    control_config_by_profile: Dict[str, ControlConfig] = {"default": base_control_config}
    if alt_control_config is not None:
        control_profiles_payload[alt_profile] = control_config_payload(
            config=alt_control_config,
            rewrite_budget_ms=control_rewrite_budget_ms,
            rewrite_budget_multiplier=control_rewrite_budget_multiplier,
        )
        control_config_by_profile[alt_profile] = alt_control_config
    control_profile_id = hashlib.sha256(
        json.dumps(control_profiles_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    tracked_files = [
        "core/response_controller.py",
        "scripts/inference_adapter.py",
        "scripts/run_proposal_alignment_eval.py",
        "scripts/run_proposal_alignment_eval_batched.py",
        "cpp/include/ResponseController.h",
        "cpp/src/ResponseController.cpp",
        "cpp/include/NPCInference.h",
        "cpp/src/NPCInference.cpp",
    ]
    code_revision = gather_code_revision_metadata(ROOT_DIR)
    tracked_hashes = collect_tracked_file_hashes(ROOT_DIR, tracked_files)
    preflight_gate_config = {
        "enabled": bool(args.preflight_operational_gate),
        "arm_id": str(args.preflight_gate_arm).strip() or "proposed_contextual_controlled",
        "max_fallback_rate": float(args.preflight_max_fallback_rate),
        "max_retry_rate": float(args.preflight_max_retry_rate),
        "min_first_pass_accept_rate": float(args.preflight_min_first_pass_accept_rate),
    }

    run_config = {
        "run_id": run_id,
        "generated_utc": utc_iso(),
        "host": args.host,
        "candidate_model": args.candidate_model,
        "baseline_model": args.baseline_model,
        "baseline_models": baseline_models,
        "baseline_arm_ids": baseline_arm_ids,
        "skip_external_baselines": bool(args.skip_external_baselines),
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "repeats": args.repeats,
        "max_scenarios": int(args.max_scenarios),
        "scenario_diversity_gate": {
            "enforce": bool(args.enforce_scenario_diversity),
            "min_template_signature_ratio": min_sig_ratio,
            "min_effective_source_n": min_eff_source_n,
        },
        "seed": args.seed,
        "timeout_s": int(args.timeout_s),
        "min_arm_success_rate": float(args.min_arm_success_rate),
        "bertscore_lang": args.bertscore_lang,
        "bertscore_model_type": str(args.bertscore_model_type),
        "bertscore_batch_size": int(args.bertscore_batch_size),
        "bertscore_cache_dir": str(args.bertscore_cache_dir),
        "disable_bertscore": bool(args.disable_bertscore),
        "target_arm": args.target_arm,
        "slice_keys": parse_list_arg(str(args.slice_keys)),
        "human_eval_file": str(args.human_eval_file),
        "human_eval_metrics": parse_list_arg(str(args.human_eval_metrics)),
        "human_eval_scale_max": float(args.human_eval_scale_max),
        "scenario_path": str(scenarios_path),
        "scenario_sha256": sha256_file(scenarios_path),
        "arms": [
            {
                "arm_id": arm.arm_id,
                "model": arm.model,
                "include_dynamic_context": arm.include_dynamic_context,
                "use_response_control": arm.use_response_control,
                "control_profile": arm.control_profile,
            }
            for arm in arms
        ],
        "response_control_config": response_control_config_payload,
        "response_control_profiles": control_profiles_payload,
        "control_alt_profile": alt_profile,
        "control_alt_arm_id": alt_arm_id if alt_control_config is not None else "",
        "control_alt_overrides": alt_overrides if alt_control_config is not None else {},
        "control_alt_overrides_file": alt_overrides_file if alt_control_config is not None else "",
        "reproducibility_lock": {
            "control_profile_id": control_profile_id,
            "code_revision": code_revision,
            "tracked_files_sha256": tracked_hashes,
        },
        "operational_preflight_gate": preflight_gate_config,
        "scenario_dependence_summary": {
            "unique_source_count": int(scenario_dependence.get("unique_source_count", 0)),
            "unique_template_signature_count": int(scenario_dependence.get("unique_template_signature_count", 0)),
            "template_signature_ratio": float(scenario_dependence.get("template_signature_ratio", 0.0) or 0.0),
            "effective_sample_size_by_source": float(
                scenario_dependence.get("effective_sample_size_by_source", float("nan"))
            ),
        },
    }
    write_json(run_dir / "run_config.json", run_config)
    write_jsonl(run_dir / "scenarios.jsonl", scenarios)
    write_json(run_dir / "scenario_dependence.json", scenario_dependence)

    hardware = gather_hardware_metadata()
    write_json(metadata_dir / "hardware.json", hardware)

    model_meta: Dict[str, Any] = {}
    for model_name in sorted({arm.model for arm in arms}):
        model_meta[model_name] = query_ollama_model(args.host, model_name)
    write_json(metadata_dir / "models.json", model_meta)

    all_responses: Dict[str, List[Dict[str, Any]]] = {arm.arm_id: [] for arm in arms}
    request_index = 0

    for arm in arms:
        print(
            f"[arm] {arm.arm_id} model={arm.model} dynamic_context={arm.include_dynamic_context} "
            f"response_control={arm.use_response_control} profile={arm.control_profile}"
        )
        for repeat_idx in range(max(1, args.repeats)):
            for scenario in scenarios:
                scenario_id = str(scenario.get("scenario_id", ""))
                request_index += 1
                prompt = build_prompt(scenario, include_dynamic_context=arm.include_dynamic_context)
                gen = generate_ollama_response(
                    host=args.host,
                    model=arm.model,
                    prompt=prompt,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    timeout_s=max(3, args.timeout_s),
                )

                record = {
                    "request_index": request_index,
                    "repeat_index": repeat_idx,
                    "arm_id": arm.arm_id,
                    "model": arm.model,
                    "scenario_id": scenario_id,
                    "include_dynamic_context": arm.include_dynamic_context,
                    "use_response_control": arm.use_response_control,
                    "timestamp_utc": utc_iso(),
                    "prompt": prompt,
                    "prompt_chars": len(prompt),
                    "reference_response": scenario.get("reference_response", ""),
                }
                record.update(gen)

                raw_response = str(record.get("response", ""))
                record["raw_response"] = raw_response

                if arm.use_response_control:
                    scenario_persona = str(scenario.get("persona", ""))
                    scenario_context = str(scenario.get("dynamic_context", "")) if arm.include_dynamic_context else ""
                    scenario_input = str(scenario.get("player_input", ""))
                    context_keywords = [str(x) for x in scenario.get("context_keywords", [])] if arm.include_dynamic_context else []
                    persona_keywords = [str(x) for x in scenario.get("persona_keywords", [])]
                    arm_profile = str(arm.control_profile or "default")
                    active_control = control_config_by_profile.get(arm_profile, base_control_config)
                    control_config = ControlConfig(
                        min_context_coverage=(
                            active_control.min_context_coverage if arm.include_dynamic_context else 0.0
                        ),
                        min_persona_coverage=active_control.min_persona_coverage,
                        min_response_tokens=active_control.min_response_tokens,
                        rewrite_temperature=active_control.rewrite_temperature,
                        rewrite_max_tokens=active_control.rewrite_max_tokens,
                        rewrite_candidates=active_control.rewrite_candidates,
                        rewrite_temperature_step=active_control.rewrite_temperature_step,
                        early_stop_on_pass=active_control.early_stop_on_pass,
                        early_stop_score=active_control.early_stop_score,
                        allow_relaxed_acceptance=active_control.allow_relaxed_acceptance,
                        relaxed_context_coverage=(
                            active_control.relaxed_context_coverage if arm.include_dynamic_context else 0.0
                        ),
                        relaxed_persona_coverage=active_control.relaxed_persona_coverage,
                        relaxed_candidate_score=active_control.relaxed_candidate_score,
                        min_rewrite_gain=active_control.min_rewrite_gain,
                        enable_rewrite=active_control.enable_rewrite,
                        allow_best_effort_rewrite=active_control.allow_best_effort_rewrite,
                        behavior_adaptation_enabled=active_control.behavior_adaptation_enabled,
                        adaptive_acceptance_enabled=active_control.adaptive_acceptance_enabled,
                        adaptive_candidate_score=active_control.adaptive_candidate_score,
                        adaptive_context_coverage=(
                            active_control.adaptive_context_coverage if arm.include_dynamic_context else 0.0
                        ),
                        adaptive_persona_coverage=active_control.adaptive_persona_coverage,
                        adaptive_high_confidence_score=active_control.adaptive_high_confidence_score,
                        adaptive_mid_confidence_score=active_control.adaptive_mid_confidence_score,
                        adaptive_high_confidence_rewrites=active_control.adaptive_high_confidence_rewrites,
                        adaptive_mid_confidence_rewrites=active_control.adaptive_mid_confidence_rewrites,
                        adaptive_low_confidence_rewrites=active_control.adaptive_low_confidence_rewrites,
                        low_confidence_retry_requires_gain=active_control.low_confidence_retry_requires_gain,
                        low_confidence_retry_min_score_gain=active_control.low_confidence_retry_min_score_gain,
                        low_confidence_retry_min_coverage_gain=(
                            active_control.low_confidence_retry_min_coverage_gain
                        ),
                        intent_risk_adaptation_enabled=active_control.intent_risk_adaptation_enabled,
                        latency_adaptation_enabled=active_control.latency_adaptation_enabled,
                        latency_relax_start_pressure=active_control.latency_relax_start_pressure,
                        latency_relax_max_delta=active_control.latency_relax_max_delta,
                        low_risk_context_relax=active_control.low_risk_context_relax,
                        low_risk_persona_relax=active_control.low_risk_persona_relax,
                        low_risk_candidate_score_relax=active_control.low_risk_candidate_score_relax,
                        high_risk_context_tighten=active_control.high_risk_context_tighten,
                        high_risk_persona_tighten=active_control.high_risk_persona_tighten,
                        high_risk_candidate_score_tighten=active_control.high_risk_candidate_score_tighten,
                        intent_focused_context_enabled=active_control.intent_focused_context_enabled,
                        intent_focus_min_keep=active_control.intent_focus_min_keep,
                        intent_focus_keep_ratio_low=active_control.intent_focus_keep_ratio_low,
                        intent_focus_keep_ratio_medium=active_control.intent_focus_keep_ratio_medium,
                        intent_focus_keep_ratio_high=active_control.intent_focus_keep_ratio_high,
                        intent_focus_min_relevance=active_control.intent_focus_min_relevance,
                        near_pass_enabled=active_control.near_pass_enabled,
                        near_pass_max_context_gap=active_control.near_pass_max_context_gap,
                        near_pass_max_persona_gap=active_control.near_pass_max_persona_gap,
                        near_pass_score_floor=active_control.near_pass_score_floor,
                        near_pass_block_high_risk=active_control.near_pass_block_high_risk,
                    )
                    scenario_tags = get_scenario_tags(scenario)
                    behavior_state = str(scenario_tags.get("behavior_state", "")).strip()

                    rewrite_meta: Dict[str, Any] = {
                        "attempts": 0,
                        "successful_attempts": 0,
                        "last_error": "",
                        "total_eval_count": 0,
                        "total_duration_ns": 0,
                        "budget_exhausted": False,
                    }
                    raw_total_duration_ns = int(record.get("total_duration_ns", 0) or 0)
                    raw_total_duration_ms = raw_total_duration_ns / 1_000_000.0
                    rewrite_budget_ms = float("inf")
                    if control_rewrite_budget_ms > 0.0:
                        rewrite_budget_ms = min(rewrite_budget_ms, control_rewrite_budget_ms)
                    if control_rewrite_budget_multiplier > 0.0 and raw_total_duration_ms > 0.0:
                        rewrite_budget_ms = min(
                            rewrite_budget_ms,
                            control_rewrite_budget_multiplier * raw_total_duration_ms,
                        )

                    def rewrite_fn(rewrite_prompt: str, rewrite_max_tokens: int, rewrite_temperature: float) -> str:
                        spent_ms = int(rewrite_meta.get("total_duration_ns", 0) or 0) / 1_000_000.0
                        if spent_ms >= rewrite_budget_ms:
                            rewrite_meta["budget_exhausted"] = True
                            return ""
                        rewrite_meta["attempts"] = int(rewrite_meta.get("attempts", 0) or 0) + 1
                        start_ns = time.perf_counter_ns()
                        rewrite_gen = generate_ollama_response(
                            host=args.host,
                            model=arm.model,
                            prompt=rewrite_prompt,
                            temperature=rewrite_temperature,
                            max_tokens=rewrite_max_tokens,
                            timeout_s=max(3, args.timeout_s),
                        )
                        ok = bool(rewrite_gen.get("ok"))
                        if ok:
                            rewrite_meta["successful_attempts"] = int(
                                rewrite_meta.get("successful_attempts", 0) or 0
                            ) + 1
                        else:
                            rewrite_meta["last_error"] = str(rewrite_gen.get("error", ""))
                        rewrite_meta["total_eval_count"] = int(rewrite_meta.get("total_eval_count", 0) or 0) + int(
                            rewrite_gen.get("eval_count", 0) or 0
                        )
                        duration_ns = int(rewrite_gen.get("total_duration_ns", 0) or 0)
                        if duration_ns <= 0:
                            duration_ns = max(0, time.perf_counter_ns() - start_ns)
                        rewrite_meta["total_duration_ns"] = int(
                            rewrite_meta.get("total_duration_ns", 0) or 0
                        ) + duration_ns
                        return str(rewrite_gen.get("response", ""))

                    control_result = control_response(
                        raw_response=raw_response,
                        persona=scenario_persona,
                        dynamic_context=scenario_context,
                        player_input=scenario_input,
                        context_keywords=context_keywords,
                        persona_keywords=persona_keywords,
                        rewrite_fn=rewrite_fn,
                        config=control_config,
                        behavior_state=behavior_state,
                        raw_latency_ms=raw_total_duration_ms,
                        timeout_s=float(args.timeout_s),
                    )

                    record["response"] = control_result.response
                    record["response_sanitized"] = (sanitize_npc_response(raw_response) != control_result.response)
                    record["response_control_source"] = control_result.source
                    record["response_control_profile"] = arm_profile
                    record["response_repaired"] = control_result.repaired
                    record["response_repair_reason"] = control_result.repair_reason
                    record["response_context_coverage"] = control_result.context_coverage
                    record["response_persona_coverage"] = control_result.persona_coverage
                    if rewrite_meta:
                        record["rewrite_attempted"] = True
                        record["rewrite_attempts"] = int(rewrite_meta.get("attempts", 0) or 0)
                        record["rewrite_successful_attempts"] = int(
                            rewrite_meta.get("successful_attempts", 0) or 0
                        )
                        record["rewrite_ok"] = bool(record["rewrite_successful_attempts"] > 0)
                        record["rewrite_error"] = str(rewrite_meta.get("last_error", ""))
                        record["rewrite_eval_count"] = int(rewrite_meta.get("total_eval_count", 0) or 0)
                        record["rewrite_total_duration_ns"] = int(rewrite_meta.get("total_duration_ns", 0) or 0)
                        record["rewrite_budget_exhausted"] = bool(rewrite_meta.get("budget_exhausted", False))
                    if control_result.source == "fallback":
                        record["response_fallback"] = True
                    if behavior_state:
                        record["behavior_state"] = behavior_state
                else:
                    sanitized = sanitize_npc_response(raw_response)
                    if sanitized:
                        record["response"] = sanitized
                        record["response_sanitized"] = (sanitized != raw_response)
                    else:
                        scenario_persona = str(scenario.get("persona", ""))
                        scenario_context = str(scenario.get("dynamic_context", "")) if arm.include_dynamic_context else ""
                        scenario_input = str(scenario.get("player_input", ""))
                        persona_keywords = [str(x) for x in scenario.get("persona_keywords", [])]
                        fallback_text = sanitize_npc_response(
                            grounded_fallback_response(
                                persona=scenario_persona,
                                dynamic_context=scenario_context,
                                player_input=scenario_input,
                                persona_keywords=persona_keywords,
                            )
                        )
                        if not fallback_text:
                            fallback_text = "I can help, but I need one concrete detail so I can respond safely and in character."
                        record["response"] = fallback_text
                        record["response_sanitized"] = True
                        record["response_fallback"] = True

                all_responses[arm.arm_id].append(record)

                state = "ok" if record.get("ok") else "error"
                preview = str(record.get("response", ""))[:70].replace("\n", " ")
                print(f"  - scenario={scenario_id} repeat={repeat_idx + 1}/{args.repeats} status={state} preview={preview}")

    for arm in arms:
        write_jsonl(responses_dir / f"{arm.arm_id}.jsonl", all_responses[arm.arm_id])

    arm_request_health: Dict[str, Dict[str, Any]] = {}
    low_success_arms: List[str] = []
    min_success_rate = max(0.0, min(1.0, float(args.min_arm_success_rate)))
    for arm in arms:
        rows = all_responses.get(arm.arm_id, [])
        total = len(rows)
        ok_count = sum(1 for row in rows if bool(row.get("ok")))
        success_rate = (ok_count / total) if total > 0 else 0.0
        arm_request_health[arm.arm_id] = {
            "total_requests": total,
            "successful_requests": ok_count,
            "success_rate": success_rate,
        }
        if total == 0 or ok_count == 0 or success_rate < min_success_rate:
            low_success_arms.append(f"{arm.arm_id}={ok_count}/{total} ({success_rate:.3f})")

    run_config["arm_request_health"] = arm_request_health
    write_json(run_dir / "run_config.json", run_config)
    if low_success_arms:
        details = "; ".join(low_success_arms)
        raise RuntimeError(
            "Insufficient successful generations for one or more arms. "
            f"Required min-arm-success-rate={min_success_rate:.2f}. Details: {details}"
        )

    operational_metrics = summarize_operational_metrics(
        responses_by_arm=all_responses,
        scenarios_by_id=scenarios_by_id,
    )
    write_json(run_dir / "operational_metrics.json", operational_metrics)
    gate_arm_id = str(args.preflight_gate_arm).strip() or "proposed_contextual_controlled"
    if gate_arm_id not in operational_metrics.get("arms", {}):
        gate_arm_id = "proposed_contextual_controlled" if "proposed_contextual_controlled" in operational_metrics.get("arms", {}) else arms[0].arm_id
    operational_gate = evaluate_operational_gate(
        operational_metrics=operational_metrics,
        arm_id=gate_arm_id,
        max_fallback_rate=float(args.preflight_max_fallback_rate),
        max_retry_rate=float(args.preflight_max_retry_rate),
        min_first_pass_accept_rate=float(args.preflight_min_first_pass_accept_rate),
    )
    write_json(run_dir / "preflight_operational_gate.json", operational_gate)
    run_config["operational_preflight_gate_result"] = operational_gate
    write_json(run_dir / "run_config.json", run_config)
    if bool(args.preflight_operational_gate) and not bool(operational_gate.get("pass")):
        details = ", ".join(
            f"{c.get('name')}={c.get('observed')} ({c.get('rule')})"
            for c in operational_gate.get("checks", [])
            if not bool(c.get("pass"))
        )
        raise RuntimeError(
            "Operational preflight gate failed before scoring. "
            f"Arm={operational_gate.get('arm_id')}. Failing checks: {details}"
        )

    all_scores: Dict[str, List[Dict[str, Any]]] = {}
    summary: Dict[str, Dict[str, Any]] = {}

    bertscore_disabled = bool(args.disable_bertscore)
    bertscore_meta: Dict[str, Any] = {"available": False, "reason": "disabled_by_flag"} if bertscore_disabled else {"available": False}
    bertscore_available_any = False

    for idx, arm in enumerate(arms):
        rows = [r for r in all_responses[arm.arm_id] if r.get("ok")]
        scored = evaluate_responses_for_arm(rows, scenarios_by_id=scenarios_by_id)

        if not bertscore_disabled:
            refs = [str(row.get("reference_response", "")) for row in scored]
            hyps = [str(row.get("response", "")) for row in scored]
            bert = compute_optional_bertscore(
                hyps,
                refs,
                lang=args.bertscore_lang,
                model_type=str(args.bertscore_model_type),
                batch_size=int(args.bertscore_batch_size),
                cache_dir=str(args.bertscore_cache_dir),
            )
            if bert.get("available"):
                bertscore_meta = {
                    "available": True,
                    "lang": bert.get("lang", args.bertscore_lang),
                    "model_type": bert.get("model_type", str(args.bertscore_model_type).strip() or "auto"),
                    "batch_size": int(bert.get("batch_size", int(args.bertscore_batch_size))),
                    "cache_dir": str(bert.get("cache_dir", str(args.bertscore_cache_dir))),
                }
                bertscore_available_any = True
                vals = bert.get("values", [])
                for row, f1 in zip(scored, vals):
                    row["bertscore_f1"] = float(f1)
            else:
                if "reason" in bert and not bertscore_available_any:
                    bertscore_meta = {"available": False, "reason": bert["reason"]}

        for row in scored:
            row["overall_quality"] = row_overall_quality(row)

        all_scores[arm.arm_id] = scored
        write_jsonl(scores_dir / f"{arm.arm_id}.jsonl", scored)
        summary[arm.arm_id] = summarize_arm_scores(
            scored_rows=scored,
            seed=args.seed + 401 * (idx + 1),
            include_bertscore=any("bertscore_f1" in row for row in scored),
        )

    write_json(run_dir / "summary.json", summary)

    available_arm_ids = {arm.arm_id for arm in arms}
    metrics_for_compare = metric_names(include_bertscore=bertscore_available_any)

    def baseline_comp_name(prefix: str, baseline_arm: str, legacy_name: str) -> str:
        if baseline_arm == "baseline_no_context":
            return legacy_name
        return f"{prefix}_vs_{baseline_arm}"

    default_target_arm = "proposed_contextual_controlled" if "proposed_contextual_controlled" in available_arm_ids else "proposed_contextual"
    target_arm_for_external = str(args.target_arm).strip()
    if target_arm_for_external not in available_arm_ids:
        target_arm_for_external = default_target_arm

    comparison_plan: List[Tuple[str, str, str]] = []
    comparison_plan.append(("proposed_vs_candidate_no_context", "proposed_contextual", "candidate_no_context"))
    for baseline_arm in baseline_arm_ids:
        comparison_plan.append(
            (
                baseline_comp_name("proposed", baseline_arm, "proposed_vs_baseline_no_context"),
                "proposed_contextual",
                baseline_arm,
            )
        )

    if "proposed_contextual_controlled" in available_arm_ids:
        comparison_plan.append(("controlled_vs_proposed_raw", "proposed_contextual_controlled", "proposed_contextual"))
        comparison_plan.append(
            ("controlled_vs_candidate_no_context", "proposed_contextual_controlled", "candidate_no_context")
        )
        for baseline_arm in baseline_arm_ids:
            comparison_plan.append(
                (
                    baseline_comp_name("controlled", baseline_arm, "controlled_vs_baseline_no_context"),
                    "proposed_contextual_controlled",
                    baseline_arm,
                )
            )

    if alt_control_config is not None and alt_arm_id in available_arm_ids:
        comparison_plan.append(("controlled_alt_vs_controlled_default", alt_arm_id, "proposed_contextual_controlled"))
        comparison_plan.append(("controlled_alt_vs_proposed_raw", alt_arm_id, "proposed_contextual"))
        comparison_plan.append(("controlled_alt_vs_candidate_no_context", alt_arm_id, "candidate_no_context"))
        for baseline_arm in baseline_arm_ids:
            comparison_plan.append((f"controlled_alt_vs_{baseline_arm}", alt_arm_id, baseline_arm))

    if target_arm_for_external in available_arm_ids:
        for baseline_arm in baseline_arm_ids:
            custom_name = f"{target_arm_for_external}_vs_{baseline_arm}"
            if all(custom_name != existing[0] for existing in comparison_plan):
                comparison_plan.append((custom_name, target_arm_for_external, baseline_arm))

    deltas: Dict[str, Any] = {}
    for comp_idx, (comp_name, target_arm, baseline_arm) in enumerate(comparison_plan):
        deltas[comp_name] = diff_against_reference(
            summary,
            target_arm,
            baseline_arm,
            metrics=metrics_for_compare,
        )
    write_json(run_dir / "delta_vs_baselines.json", deltas)

    scored_by_arm = {arm.arm_id: all_scores.get(arm.arm_id, []) for arm in arms}
    paired_deltas: Dict[str, Any] = {}
    win_rates: Dict[str, Any] = {}
    for comp_idx, (comp_name, target_arm, baseline_arm) in enumerate(comparison_plan):
        paired_deltas[comp_name] = paired_metric_deltas(
            scored_by_arm,
            target_arm,
            baseline_arm,
            seed=args.seed + 5001 + 997 * (comp_idx + 1),
            metrics=metrics_for_compare,
            scenarios_by_id=scenarios_by_id,
        )
        win_rates[comp_name] = paired_metric_win_rates(
            scored_by_arm,
            target_arm,
            baseline_arm,
            seed=args.seed + 8001 + 991 * (comp_idx + 1),
            metrics=metrics_for_compare,
        )
    write_json(run_dir / "paired_delta_significance.json", paired_deltas)
    write_json(run_dir / "win_rates.json", win_rates)

    multi_turn_contradictions = analyze_multi_turn_contradictions(
        scored_by_arm=scored_by_arm,
        scenarios_by_id=scenarios_by_id,
    )
    write_json(run_dir / "multi_turn_contradictions.json", multi_turn_contradictions)

    slice_keys = parse_list_arg(str(args.slice_keys))
    slice_summary = summarize_scores_by_slice(
        scored_by_arm=scored_by_arm,
        scenarios_by_id=scenarios_by_id,
        slice_keys=slice_keys,
        seed=args.seed + 12001,
        include_bertscore=bertscore_available_any,
    )
    write_json(run_dir / "slice_summary.json", slice_summary)

    human_eval_summary: Optional[Dict[str, Any]] = None
    human_eval_path = Path(str(args.human_eval_file).strip()) if str(args.human_eval_file).strip() else None
    if human_eval_path:
        if not human_eval_path.exists():
            raise FileNotFoundError(f"Human evaluation file not found: {human_eval_path}")
        human_rows = read_human_eval_rows(human_eval_path)
        human_metrics = parse_list_arg(str(args.human_eval_metrics))
        human_eval_summary = analyze_human_eval(
            rows=human_rows,
            metrics=human_metrics,
            scale_max=float(args.human_eval_scale_max),
            target_arm=target_arm_for_external,
            baseline_arms=baseline_arm_ids,
            seed=args.seed + 14001,
        )
        write_json(run_dir / "human_eval_summary.json", human_eval_summary)
        render_human_eval_report(run_dir / "human_eval_report.md", human_eval_summary)

    error_analysis: Dict[str, Any] = {}
    for arm in arms:
        error_analysis[arm.arm_id] = analyze_errors(all_scores.get(arm.arm_id, []))
    write_json(run_dir / "error_analysis.json", error_analysis)

    write_json(
        run_dir / "comparison_plan.json",
        {
            "target_arm_for_external": target_arm_for_external,
            "baseline_arm_ids": baseline_arm_ids,
            "control_alt_arm_id": alt_arm_id if alt_control_config is not None else "",
            "control_alt_profile": alt_profile if alt_control_config is not None else "none",
            "comparisons": [
                {"comparison_id": comp_name, "target_arm": target_arm, "baseline_arm": baseline_arm}
                for comp_name, target_arm, baseline_arm in comparison_plan
            ],
            "metrics": metrics_for_compare,
        },
    )

    render_report(
        output_path=run_dir / "report.md",
        run_id=run_id,
        scenarios_path=Path(run_dir / "scenarios.jsonl"),
        arms=arms,
        summary=summary,
        deltas=deltas,
        paired_deltas=paired_deltas,
        bertscore_meta=bertscore_meta,
        win_rates=win_rates,
        slice_summary=slice_summary,
        human_eval=human_eval_summary,
        operational_metrics=operational_metrics,
        scenario_dependence=scenario_dependence,
        multi_turn_contradictions=multi_turn_contradictions,
    )

    print(f"\nPublished proposal artifact bundle: {run_dir}")
    print("Generated files:")
    print(f"  - {run_dir / 'summary.json'}")
    print(f"  - {run_dir / 'delta_vs_baselines.json'}")
    print(f"  - {run_dir / 'paired_delta_significance.json'}")
    print(f"  - {run_dir / 'win_rates.json'}")
    print(f"  - {run_dir / 'slice_summary.json'}")
    print(f"  - {run_dir / 'error_analysis.json'}")
    print(f"  - {run_dir / 'operational_metrics.json'}")
    print(f"  - {run_dir / 'preflight_operational_gate.json'}")
    print(f"  - {run_dir / 'scenario_dependence.json'}")
    print(f"  - {run_dir / 'multi_turn_contradictions.json'}")
    if human_eval_summary is not None:
        print(f"  - {run_dir / 'human_eval_summary.json'}")
        print(f"  - {run_dir / 'human_eval_report.md'}")
    print(f"  - {run_dir / 'report.md'}")


if __name__ == "__main__":
    main()

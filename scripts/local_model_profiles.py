#!/usr/bin/env python3
"""Shared local-model profile helpers for laptop-safe benchmark baselines."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Set, Tuple

import requests


BASELINE_PROFILES: Dict[str, List[str]] = {
    "none": [],
    # Recommended for 4GB-class laptop setups with quantized Ollama tags.
    "ultra_safe": [
        "phi3:mini",
        "gemma2:2b",
    ],
    "laptop_safe": [
        "phi3:mini",
        "phi3:latest",
        "gemma2:2b",
        "qwen2.5:3b-instruct",
    ],
    # Slightly broader comparison set; may be slower on 4GB but still feasible when quantized.
    "laptop_extended": [
        "phi3:mini",
        "phi3:latest",
        "gemma2:2b",
        "qwen2.5:3b-instruct",
        "llama3.2",
    ],
}


def baseline_profile_choices() -> List[str]:
    return sorted(BASELINE_PROFILES.keys())


def parse_model_csv(raw: str) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for token in str(raw or "").replace(";", ",").split(","):
        t = token.strip()
        if not t:
            continue
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


def merge_unique_models(*groups: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for group in groups:
        for token in group:
            t = str(token).strip()
            if not t:
                continue
            key = t.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(t)
    return out


def resolve_baseline_models(raw_baselines: str, profile: str) -> List[str]:
    parsed = parse_model_csv(raw_baselines)
    profile_key = str(profile or "none").strip().lower()
    profile_models = BASELINE_PROFILES.get(profile_key, [])
    return merge_unique_models(parsed, profile_models)


def format_model_csv(models: Sequence[str]) -> str:
    return ",".join(merge_unique_models(models))


def fetch_ollama_models(host: str, timeout_s: int = 10) -> Set[str]:
    resp = requests.get(f"{host}/api/tags", timeout=timeout_s)
    resp.raise_for_status()
    payload = resp.json()
    return {
        str(item.get("name", "")).strip()
        for item in payload.get("models", [])
        if str(item.get("name", "")).strip()
    }


def split_available_missing(requested: Sequence[str], installed: Set[str]) -> Tuple[List[str], List[str]]:
    avail: List[str] = []
    missing: List[str] = []
    for model in merge_unique_models(requested):
        if model in installed:
            avail.append(model)
        else:
            missing.append(model)
    return avail, missing

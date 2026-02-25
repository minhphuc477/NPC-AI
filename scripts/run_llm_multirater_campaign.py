#!/usr/bin/env python3
"""Run an LLM multi-rater evaluation campaign and emit human-eval compatible CSV."""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import requests


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def choose_one_response_per_scenario(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    selected: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        sid = str(row.get("scenario_id", "")).strip()
        if not sid:
            continue
        key = (int(row.get("repeat_index", 999999) or 999999), int(row.get("request_index", 999999999) or 999999999))
        prev = selected.get(sid)
        if prev is None:
            selected[sid] = dict(row)
            selected[sid]["_k"] = key
            continue
        if key < prev.get("_k", (999999, 999999999)):
            selected[sid] = dict(row)
            selected[sid]["_k"] = key
    for sid in list(selected.keys()):
        selected[sid].pop("_k", None)
    return selected


def parse_annotators(raw: str) -> List[Tuple[str, str, float]]:
    # Format: "model|profile|temperature,model|profile|temperature"
    out: List[Tuple[str, str, float]] = []
    for token in [x.strip() for x in str(raw).split(",") if x.strip()]:
        parts = [p.strip() for p in token.split("|")]
        if len(parts) == 1:
            out.append((parts[0], "balanced", 0.0))
            continue
        if len(parts) == 2:
            out.append((parts[0], parts[1] or "balanced", 0.0))
            continue
        model = parts[0]
        profile = parts[1] or "balanced"
        try:
            temp = float(parts[2])
        except Exception:
            temp = 0.0
        out.append((model, profile, temp))
    dedup: List[Tuple[str, str, float]] = []
    seen = set()
    for model, profile, temp in out:
        key = (model.lower(), profile.lower(), round(float(temp), 3))
        if key in seen:
            continue
        seen.add(key)
        dedup.append((model, profile, float(temp)))
    return dedup


def build_profile_instruction(profile: str) -> str:
    p = profile.strip().lower()
    if p == "grounding_strict":
        return (
            "Prioritize grounding to provided context and player input. "
            "Penalize unsupported claims and generic drift."
        )
    if p == "persona_style":
        return (
            "Prioritize persona consistency and role-play voice. "
            "Penalize style drift and out-of-character responses."
        )
    if p == "naturalness":
        return (
            "Prioritize naturalness and conversational quality while keeping context fit."
        )
    return (
        "Balance context relevance, persona consistency, and naturalness equally."
    )


def build_judge_prompt(
    scenario: Dict[str, Any],
    arms: Sequence[str],
    responses_by_arm: Dict[str, str],
    profile: str,
) -> str:
    persona = str(scenario.get("persona", "")).strip()
    dynamic_context = str(scenario.get("dynamic_context", "")).strip()
    player_input = str(scenario.get("player_input", "")).strip()
    profile_instruction = build_profile_instruction(profile)

    lines: List[str] = []
    lines.append("You are an expert NPC dialogue evaluator.")
    lines.append(profile_instruction)
    lines.append("Rate each arm from 1 to 5 on:")
    lines.append("- context_relevance")
    lines.append("- persona_consistency")
    lines.append("- naturalness")
    lines.append("- overall_quality")
    lines.append("")
    lines.append("Scenario:")
    lines.append(f"Persona: {persona}")
    lines.append(f"Dynamic Context: {dynamic_context}")
    lines.append(f"Player Input: {player_input}")
    lines.append("")
    lines.append("Candidate Responses:")
    for arm in arms:
        lines.append(f"[{arm}] {responses_by_arm.get(arm, '')}")
    lines.append("")
    lines.append("Output format requirement (no extra text):")
    lines.append("One line per arm, tab-separated with exactly 6 fields:")
    lines.append("arm_id<TAB>context_relevance(1-5)<TAB>persona_consistency(1-5)<TAB>naturalness(1-5)<TAB>overall_quality(1-5)<TAB>short note")
    lines.append("Use each provided arm_id exactly once.")
    lines.append("Do not invent or rename arm IDs. Do not add markdown/code fences.")
    return "\n".join(lines)


def build_single_arm_judge_prompt(
    scenario: Dict[str, Any],
    arm_id: str,
    response: str,
    profile: str,
) -> str:
    profile_instruction = build_profile_instruction(profile)
    persona = str(scenario.get("persona", "")).strip()
    dynamic_context = str(scenario.get("dynamic_context", "")).strip()
    player_input = str(scenario.get("player_input", "")).strip()
    lines: List[str] = []
    lines.append("You are an expert NPC dialogue evaluator.")
    lines.append(profile_instruction)
    lines.append("Rate the single response on 1-5 for context_relevance, persona_consistency, naturalness, overall_quality.")
    lines.append("")
    lines.append("Scenario:")
    lines.append(f"Persona: {persona}")
    lines.append(f"Dynamic Context: {dynamic_context}")
    lines.append(f"Player Input: {player_input}")
    lines.append("")
    lines.append(f"Arm ID: {arm_id}")
    lines.append(f"Response: {response}")
    lines.append("")
    lines.append("Output one line only, tab-separated:")
    lines.append("arm_id<TAB>context_relevance<TAB>persona_consistency<TAB>naturalness<TAB>overall_quality<TAB>short note")
    lines.append("Use exactly the same arm_id.")
    return "\n".join(lines)


def ollama_generate_text(
    host: str,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout_s: int,
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": float(temperature),
            "num_predict": int(max_tokens),
        },
    }
    resp = requests.post(f"{host}/api/generate", json=payload, timeout=timeout_s)
    resp.raise_for_status()
    data = resp.json()
    text = str(data.get("response", "")).strip()
    if not text:
        raise RuntimeError("empty_response")
    return text


def normalize_arm_id(token: str, required_arms: Sequence[str]) -> str:
    raw = str(token or "").strip().strip("[]").strip(":")
    if not raw:
        return ""
    arm_set = {a.strip() for a in required_arms}
    if raw in arm_set:
        return raw
    canon_raw = re.sub(r"[^a-z0-9]+", "", raw.lower())
    for arm in required_arms:
        if canon_raw == re.sub(r"[^a-z0-9]+", "", arm.lower()):
            return arm
    for arm in required_arms:
        if canon_raw and canon_raw in re.sub(r"[^a-z0-9]+", "", arm.lower()):
            return arm
    # Fuzzy fallback for minor model typos (e.g. baseline_no03_latest).
    match = difflib.get_close_matches(raw, list(required_arms), n=1, cutoff=0.55)
    if match:
        return match[0]
    return ""


def parse_ratings_text(raw_text: str, required_arms: Sequence[str]) -> Dict[str, Any]:
    text = (raw_text or "").replace("\\t", "\t").strip()
    if not text:
        return {"ratings": []}

    # Try strict JSON first.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and isinstance(obj.get("ratings"), list):
            return obj
    except Exception:
        pass

    # Fallback: extract first JSON object.
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict) and isinstance(obj.get("ratings"), list):
                return obj
        except Exception:
            pass

    # Final fallback: parse tab-separated lines.
    arm_set = {a.strip() for a in required_arms}
    ratings: List[Dict[str, Any]] = []
    seen_arms = set()
    for line in text.splitlines():
        row = line.strip()
        if not row:
            continue
        parts = [p.strip() for p in row.split("\t")]
        if len(parts) < 5:
            # Try pipe-separated recovery.
            parts = [p.strip() for p in row.split("|")]
        if len(parts) < 5:
            continue
        arm_id = normalize_arm_id(parts[0], required_arms)
        if not arm_id:
            # Try an "arm:score..." format.
            left = parts[0].split(":", 1)[0].strip()
            arm_id = normalize_arm_id(left, required_arms)
        if not arm_id or arm_id not in arm_set or arm_id in seen_arms:
            continue
        note = parts[5] if len(parts) > 5 else ""
        seen_arms.add(arm_id)
        ratings.append(
            {
                "arm_id": arm_id,
                "context_relevance": parts[1],
                "persona_consistency": parts[2],
                "naturalness": parts[3],
                "overall_quality": parts[4],
                "notes": note,
            }
        )

    # Fallback: score extraction from free-form per-line output.
    if len(ratings) < len(required_arms):
        rated = {str(r.get("arm_id", "")).strip() for r in ratings if str(r.get("arm_id", "")).strip()}
        for line in text.splitlines():
            row = line.strip()
            if not row:
                continue
            arm_id = ""
            for arm in required_arms:
                if arm in row:
                    arm_id = arm
                    break
            if not arm_id:
                head = row.split(":", 1)[0]
                arm_id = normalize_arm_id(head, required_arms)
            if not arm_id or arm_id in rated:
                continue
            nums = re.findall(r"-?\d+(?:\.\d+)?", row)
            if len(nums) < 4:
                continue
            ratings.append(
                {
                    "arm_id": arm_id,
                    "context_relevance": nums[0],
                    "persona_consistency": nums[1],
                    "naturalness": nums[2],
                    "overall_quality": nums[3],
                    "notes": row[:400],
                }
            )
            rated.add(arm_id)
    return {"ratings": ratings}


def sanitize_score(value: Any) -> int:
    try:
        v = int(round(float(value)))
    except Exception:
        v = 3
    return max(1, min(5, v))


def iter_rating_rows(
    scenario_id: str,
    annotator_id: str,
    payload: Dict[str, Any],
    required_arms: Sequence[str],
) -> Iterable[Dict[str, Any]]:
    ratings = payload.get("ratings", [])
    by_arm: Dict[str, Dict[str, Any]] = {}
    if isinstance(ratings, list):
        for row in ratings:
            if not isinstance(row, dict):
                continue
            arm = str(row.get("arm_id", "")).strip()
            if arm:
                by_arm[arm] = row

    for arm in required_arms:
        row = by_arm.get(arm, {})
        yield {
            "scenario_id": scenario_id,
            "arm_id": arm,
            "annotator_id": annotator_id,
            "context_relevance": sanitize_score(row.get("context_relevance", 3)),
            "persona_consistency": sanitize_score(row.get("persona_consistency", 3)),
            "naturalness": sanitize_score(row.get("naturalness", 3)),
            "overall_quality": sanitize_score(row.get("overall_quality", 3)),
            "notes": str(row.get("notes", ""))[:400],
        }


def to_ratings_map(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in payload.get("ratings", []) if isinstance(payload.get("ratings", []), list) else []:
        if isinstance(row, dict):
            arm = str(row.get("arm_id", "")).strip()
            if arm:
                out[arm] = row
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM multi-rater campaign and output human-eval CSV.")
    parser.add_argument("--run-dir", required=True, help="Proposal run dir (artifacts/proposal/<run_id>)")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument(
        "--annotators",
        default="phi3:mini|balanced|0.0,phi3:latest|grounding_strict|0.0,phi3:mini|persona_style|0.15",
        help="Comma-separated: model|profile|temperature",
    )
    parser.add_argument("--arms", default="", help="Optional comma-separated arm IDs. Default: all run_config arms.")
    parser.add_argument("--max-scenarios", type=int, default=0, help="Optional cap; 0 means all.")
    parser.add_argument("--scenario-offset", type=int, default=0, help="Process scenarios starting from this offset.")
    parser.add_argument("--scenario-limit", type=int, default=0, help="Process at most this many scenarios after offset.")
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-retries", type=int, default=1, help="Retries for incomplete judge output.")
    parser.add_argument("--cache-dir", default=".cache/llm_multirater")
    parser.add_argument("--output-csv", default="")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    scenarios = read_jsonl(run_dir / "scenarios.jsonl")
    run_config = read_json(run_dir / "run_config.json")
    responses_dir = run_dir / "responses"

    all_arms = [str(a.get("arm_id", "")).strip() for a in run_config.get("arms", []) if str(a.get("arm_id", "")).strip()]
    if str(args.arms).strip():
        requested = [x.strip() for x in str(args.arms).split(",") if x.strip()]
        arms = [a for a in requested if a in all_arms]
    else:
        arms = list(all_arms)
    if not arms:
        raise RuntimeError("No valid arms selected.")

    per_arm_rows: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for arm in arms:
        rows = read_jsonl(responses_dir / f"{arm}.jsonl")
        per_arm_rows[arm] = choose_one_response_per_scenario(rows)

    scenarios_by_id = {str(s.get("scenario_id", "")).strip(): s for s in scenarios if str(s.get("scenario_id", "")).strip()}
    common_sids = sorted(
        sid for sid in scenarios_by_id.keys() if all(sid in per_arm_rows.get(arm, {}) for arm in arms)
    )
    if int(args.max_scenarios) > 0:
        rng = random.Random(int(args.seed))
        rng.shuffle(common_sids)
        common_sids = sorted(common_sids[: int(args.max_scenarios)])
    if int(args.scenario_offset) > 0:
        common_sids = common_sids[int(args.scenario_offset) :]
    if int(args.scenario_limit) > 0:
        common_sids = common_sids[: int(args.scenario_limit)]

    annotators = parse_annotators(args.annotators)
    if not annotators:
        raise RuntimeError("No annotators configured.")

    cache_root = Path(args.cache_dir) / run_dir.name
    out_csv = Path(args.output_csv) if str(args.output_csv).strip() else run_dir / "human_eval_llm_multirater.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "scenario_id",
        "arm_id",
        "annotator_id",
        "context_relevance",
        "persona_consistency",
        "naturalness",
        "overall_quality",
        "notes",
    ]

    total_calls = 0
    cache_hits = 0
    failures = 0
    rows_out: List[Dict[str, Any]] = []

    for ann_idx, (model, profile, temperature) in enumerate(annotators):
        annotator_id = f"{model.replace(':', '_')}__{profile}__t{temperature:.2f}"
        for sid in common_sids:
            scenario = scenarios_by_id[sid]
            responses = {arm: str(per_arm_rows[arm][sid].get("response", "")) for arm in arms}
            prompt = build_judge_prompt(scenario, arms=arms, responses_by_arm=responses, profile=profile)

            cache_file = cache_root / annotator_id / f"{sid}.json"
            payload: Dict[str, Any] | None = None
            if cache_file.exists():
                try:
                    payload = read_json(cache_file)
                    cache_hits += 1
                except Exception:
                    payload = None

            if payload is None:
                total_calls += 1
                try:
                    raw_text = ollama_generate_text(
                        host=args.host,
                        model=model,
                        prompt=prompt,
                        temperature=float(temperature),
                        max_tokens=int(args.max_tokens),
                        timeout_s=int(args.timeout_s),
                    )
                    payload = parse_ratings_text(raw_text, required_arms=arms)
                    ratings_map = to_ratings_map(payload)
                    missing = [a for a in arms if a not in ratings_map]

                    retries = 0
                    while missing and retries < int(args.max_retries):
                        retries += 1
                        retry_prompt = (
                            "You returned incomplete ratings. Re-rate all required arms exactly once.\n"
                            f"Required arms: {', '.join(arms)}\n\n"
                            + prompt
                        )
                        retry_raw = ollama_generate_text(
                            host=args.host,
                            model=model,
                            prompt=retry_prompt,
                            temperature=float(temperature),
                            max_tokens=int(args.max_tokens),
                            timeout_s=int(args.timeout_s),
                        )
                        retry_payload = parse_ratings_text(retry_raw, required_arms=arms)
                        retry_map = to_ratings_map(retry_payload)
                        if retry_map:
                            ratings_map.update(retry_map)
                        missing = [a for a in arms if a not in ratings_map]

                    # Targeted fallback: rate only missing arms individually.
                    for miss_arm in missing:
                        single_prompt = build_single_arm_judge_prompt(
                            scenario=scenario,
                            arm_id=miss_arm,
                            response=responses.get(miss_arm, ""),
                            profile=profile,
                        )
                        try:
                            single_raw = ollama_generate_text(
                                host=args.host,
                                model=model,
                                prompt=single_prompt,
                                temperature=float(temperature),
                                max_tokens=min(int(args.max_tokens), 192),
                                timeout_s=int(args.timeout_s),
                            )
                            single_payload = parse_ratings_text(single_raw, required_arms=[miss_arm])
                            single_map = to_ratings_map(single_payload)
                            if miss_arm in single_map:
                                ratings_map[miss_arm] = single_map[miss_arm]
                        except Exception:
                            pass

                    payload = {"ratings": [ratings_map[a] for a in arms if a in ratings_map]}
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    write_json(cache_file, payload)
                except Exception:
                    failures += 1
                    payload = {"ratings": []}
                time.sleep(0.01)

            rows_out.extend(iter_rating_rows(scenario_id=sid, annotator_id=annotator_id, payload=payload, required_arms=arms))

    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    manifest = {
        "run_dir": str(run_dir),
        "output_csv": str(out_csv),
        "arms": arms,
        "scenario_count": len(common_sids),
        "annotators": [
            {"model": model, "profile": profile, "temperature": temperature}
            for model, profile, temperature in annotators
        ],
        "total_rows": len(rows_out),
        "api_calls": total_calls,
        "cache_hits": cache_hits,
        "failures": failures,
        "cache_root": str(cache_root),
    }
    write_json(out_csv.with_suffix(".manifest.json"), manifest)

    print(f"LLM multi-rater CSV: {out_csv}")
    print(f"Rows: {len(rows_out)}")
    print(f"Scenario count: {len(common_sids)}")
    print(f"Annotators: {len(annotators)}")
    print(f"API calls: {total_calls} | cache hits: {cache_hits} | failures: {failures}")


if __name__ == "__main__":
    main()

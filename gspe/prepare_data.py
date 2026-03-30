#!/usr/bin/env python3
"""Prepare GSPE training data from existing project artifacts."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from gspe.state_codec import FIELD_VOCABS, normalize_token

SEED = 42

BEHAVIOR_TO_ALERT: Dict[str, str] = {
    "negotiating": "investigating",
    "detained": "combat",
    "assisting": "patrol",
    "observing": "investigating",
    "researching": "idle",
    "investigating": "investigating",
    "patrolling": "patrol",
    "crafting": "idle",
    "guarding": "patrol",
    "quest_handoff": "idle",
    "recovery": "idle",
    "combat_ready": "combat",
}

BEHAVIOR_TO_SUSPICION: Dict[str, int] = {
    "negotiating": 3,
    "detained": 4,
    "assisting": 1,
    "observing": 2,
    "researching": 0,
    "investigating": 3,
    "patrolling": 1,
    "crafting": 0,
    "guarding": 1,
    "quest_handoff": 0,
    "recovery": 1,
    "combat_ready": 4,
}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_kv_context(raw: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for chunk in str(raw or "").split(";"):
        token = chunk.strip()
        if not token or "=" not in token:
            continue
        key, value = token.split("=", 1)
        out[key.strip().lower()] = value.strip()
    return out


def keep_vocab_token(field: str, value: object, fallback: str) -> str:
    token = normalize_token(value)
    if token in FIELD_VOCABS[field]:
        return token
    return fallback


def infer_game_state(record: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(record.get("game_state"), dict):
        raw = dict(record["game_state"])
    else:
        raw = {}

    if not raw and isinstance(record.get("dynamic_context"), str):
        dynamic = parse_kv_context(record.get("dynamic_context", ""))
        if "behaviortreestate" in dynamic:
            raw["behavior_state"] = dynamic.get("behaviortreestate")
        if "location" in dynamic:
            raw["location"] = dynamic.get("location")
        if "timeofday" in dynamic:
            raw["time_of_day"] = dynamic.get("timeofday")
        if "recentevent" in dynamic:
            raw["recent_event"] = dynamic.get("recentevent")
        if "weather" in dynamic:
            raw["weather"] = dynamic.get("weather")

    tags = record.get("scenario_tags", {}) if isinstance(record.get("scenario_tags"), dict) else {}
    behavior = normalize_token(
        raw.get("behavior_state")
        or record.get("behavior_state")
        or tags.get("behavior_state")
        or "patrolling"
    )

    location = keep_vocab_token(
        "location",
        raw.get("location", record.get("location", "unknown")),
        fallback="unknown",
    )
    alert_state = keep_vocab_token(
        "alert_state",
        raw.get("alert_state", BEHAVIOR_TO_ALERT.get(behavior, "patrol")),
        fallback="patrol",
    )
    suspicion = raw.get("suspicion_level", BEHAVIOR_TO_SUSPICION.get(behavior, 1))
    try:
        suspicion = int(suspicion)
    except Exception:
        suspicion = 1
    suspicion = max(0, min(4, suspicion))

    game_state = {
        "alert_state": alert_state,
        "location": location,
        "behavior_state": keep_vocab_token("behavior_state", behavior, fallback="patrolling"),
        "suspicion_level": str(suspicion),
        "time_of_day": keep_vocab_token("time_of_day", raw.get("time_of_day", "day"), fallback="day"),
        "weather": keep_vocab_token("weather", raw.get("weather", "clear"), fallback="clear"),
        "nearby_threat": bool(raw.get("nearby_threat", behavior in {"detained", "combat_ready"})),
        "active_quest": bool(raw.get("active_quest", behavior in {"quest_handoff", "investigating"})),
        "recent_event": keep_vocab_token("recent_event", raw.get("recent_event", "none"), fallback="none"),
    }
    return game_state


def resolve_persona(record: Dict[str, Any]) -> str:
    return str(
        record.get("persona")
        or record.get("character")
        or record.get("npc_name")
        or "NPC"
    ).strip()


def resolve_player_input(record: Dict[str, Any]) -> str:
    return str(
        record.get("player_input")
        or record.get("input")
        or record.get("user_message")
        or ""
    ).strip()


def resolve_response(record: Dict[str, Any]) -> str:
    return str(
        record.get("response")
        or record.get("chosen")
        or record.get("reference_response")
        or record.get("controlled_response")
        or record.get("output")
        or ""
    ).strip()


def resolve_passages(record: Dict[str, Any]) -> List[str]:
    values = record.get("retrieved_passages") or record.get("passages") or []
    if not isinstance(values, list):
        return []
    out = [str(item).strip() for item in values if str(item).strip()]
    return out[:3]


def convert_record(record: Dict[str, Any], source: str) -> Optional[Dict[str, Any]]:
    response = resolve_response(record)
    player_input = resolve_player_input(record)
    if len(response) < 8 or not player_input:
        return None
    return {
        "game_state": infer_game_state(record),
        "persona": resolve_persona(record),
        "player_input": player_input,
        "response": response,
        "retrieved_passages": resolve_passages(record),
        "source": source,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build GSPE training corpus from existing datasets.")
    parser.add_argument("--sft-corpus", default="data/train_full.jsonl")
    parser.add_argument("--benchmark", default="data/proposal_eval_scenarios_large_v2.jsonl")
    parser.add_argument("--dpo-pairs", default="storage/artifacts/datasets/dpo_data/dpo_pairs_combined.jsonl")
    parser.add_argument("--out", default="storage/artifacts/datasets/gspe/gspe_training.jsonl")
    parser.add_argument("--min-response-len", type=int, default=15)
    args = parser.parse_args()

    random.seed(SEED)

    inputs = [
        ("sft", Path(args.sft_corpus)),
        ("benchmark", Path(args.benchmark)),
        ("dpo", Path(args.dpo_pairs)),
    ]

    rows: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}
    for source, path in inputs:
        if not path.exists():
            continue
        for record in read_jsonl(path):
            converted = convert_record(record, source=source)
            if converted is None:
                continue
            if len(str(converted["response"])) < int(args.min_response_len):
                continue
            rows.append(converted)
            counts[source] = counts.get(source, 0) + 1

    if not rows:
        raise RuntimeError("No GSPE rows were produced. Check input files.")

    random.shuffle(rows)
    write_jsonl(Path(args.out), rows)

    summary = {
        "total_rows": len(rows),
        "source_counts": counts,
        "output_path": str(Path(args.out)),
    }
    summary_path = Path(args.out).with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved {len(rows)} GSPE records to {args.out}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

"""Lightweight game-state vocabulary + encoding helpers for GSPE.

This module intentionally avoids heavy ML imports so runtime handlers can
encode game state without importing torch/transformers.
"""

from __future__ import annotations

from typing import Dict, List

FIELD_VOCABS: Dict[str, List[str]] = {
    "alert_state": ["patrol", "investigating", "combat", "idle"],
    "location": ["city_gate", "shrine", "harbor", "market", "hall", "wilderness", "unknown"],
    "behavior_state": [
        "patrolling",
        "negotiating",
        "detained",
        "assisting",
        "observing",
        "researching",
        "investigating",
        "crafting",
        "guarding",
        "quest_handoff",
        "recovery",
        "combat_ready",
    ],
    "suspicion_level": ["0", "1", "2", "3", "4"],
    "time_of_day": ["dawn", "day", "dusk", "night"],
    "weather": ["clear", "rain", "storm", "fog"],
    "nearby_threat": ["false", "true"],
    "active_quest": ["false", "true"],
    "recent_event": ["none", "combat", "theft", "discovery", "arrest"],
}

_TOKEN_TO_ID: Dict[str, Dict[str, int]] = {
    field: {value: idx for idx, value in enumerate(values)}
    for field, values in FIELD_VOCABS.items()
}


def normalize_token(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).strip().lower().replace(" ", "_")


def encode_game_state(gs: Dict[str, object]) -> Dict[str, int]:
    """Map raw game-state values to GSPE vocab IDs."""
    out: Dict[str, int] = {}
    for field, vocab_map in _TOKEN_TO_ID.items():
        raw_value = gs.get(field, None)
        if raw_value is None:
            out[field] = 0
            continue
        token = normalize_token(raw_value)
        out[field] = vocab_map.get(token, 0)
    return out

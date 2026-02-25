#!/usr/bin/env python3
"""Generate a larger proposal-evaluation scenario set from seed scenarios."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

TOKEN_RE = re.compile(r"[a-z0-9']+")

LOCATION_POOLS: Dict[str, List[str]] = {
    "security": ["Village Gate", "East Watch Gate", "Inner Citadel Checkpoint", "North Barracks Entry"],
    "commerce": ["Marketplace", "Harbor Market", "Guild Bazaar", "South Trade Lane"],
    "knowledge": ["Ancient Library", "Archive Annex", "Scriptorium Hall", "University Records Room"],
    "craft": ["Blacksmith Forge", "Armory Workshop", "Tanner Yard", "Engineers' Foundry"],
    "detention": ["Castle Dungeon", "Holding Cells", "Interrogation Hall", "Watch Tower Cellblock"],
    "wilderness": ["Forest Clearing", "Moonlit Grove", "Old Well Trail", "Ravine Shrine"],
    "medical": ["Healer Hut", "Field Clinic", "Temple Infirmary", "Herbal Ward"],
    "other": ["Town Square", "Riverside Camp", "Granary Yard", "Chapel Courtyard"],
}

BEHAVIOR_POOL = [
    "Patrolling",
    "Investigating",
    "Negotiating",
    "Guarding",
    "Assisting",
    "Observing",
    "Researching",
    "Trading",
    "TreatingPatient",
    "RitualPreparation",
]

BEHAVIOR_BY_ARCHETYPE: Dict[str, List[str]] = {
    "gatekeeper": ["Patrolling", "Guarding", "Investigating", "Observing"],
    "merchant": ["Trading", "Negotiating", "Observing", "Assisting"],
    "healer": ["TreatingPatient", "Assisting", "Observing", "Researching"],
    "blacksmith": ["Forging", "Assisting", "Observing", "Investigating"],
    "scholar": ["Researching", "Observing", "Investigating", "Assisting"],
    "captain": ["Investigating", "Guarding", "Patrolling", "Negotiating"],
    "prisoner": ["Detained", "Negotiating", "Observing", "Assisting"],
    "witch": ["RitualPreparation", "Researching", "Observing", "Assisting"],
}

NEARBY_POOL = [
    "Player and two anxious villagers",
    "A nervous witness and one city guard",
    "An apprentice taking notes",
    "A militia recruit waiting for orders",
    "A courier with sealed dispatches",
    "Two traders arguing over supplies",
    "A sleeping guard at corridor end",
    "Flickering spirit lights and ravens",
]

EVENT_SUFFIXES = [
    "A messenger brought conflicting reports from the north road.",
    "Rain damaged several supply crates before dawn.",
    "A forged document was discovered during inspection.",
    "A loud alarm forced a temporary lockdown order.",
    "A trusted witness changed testimony after questioning.",
    "An injured scout arrived with urgent battlefield notes.",
    "A missing ledger page raised suspicion of tampering.",
    "Rumors spread that an impostor used a stolen seal.",
]

PLAYER_PREFIXES = [
    "",
    "Please, this is urgent: ",
    "Listen carefully, ",
    "I need an answer now: ",
    "No delays, ",
    "For the record, ",
]

PLAYER_SUFFIXES = [
    "",
    " I can explain after this.",
    " We are running out of time.",
    " I need a direct answer.",
    " Treat this as priority.",
]


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
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_dynamic_context_map(dynamic_context: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for chunk in [x.strip() for x in str(dynamic_context).split(";") if x.strip()]:
        if "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        out[str(key).strip().lower()] = str(value).strip()
    return out


def format_dynamic_context(parts: Dict[str, str]) -> str:
    return (
        f"BehaviorTreeState={parts.get('behaviortreestate', '').strip()}; "
        f"Location={parts.get('location', '').strip()}; "
        f"NearbyEntity={parts.get('nearbyentity', '').strip()}; "
        f"RecentEvent={parts.get('recentevent', '').strip()}"
    )


def infer_location_type(location: str) -> str:
    lowered = location.lower()
    if any(k in lowered for k in ("gate", "wall", "checkpoint", "barracks")):
        return "security"
    if any(k in lowered for k in ("market", "bazaar", "trade", "shop")):
        return "commerce"
    if any(k in lowered for k in ("library", "archive", "scriptorium", "records")):
        return "knowledge"
    if any(k in lowered for k in ("forge", "armory", "workshop", "foundry")):
        return "craft"
    if any(k in lowered for k in ("dungeon", "cell", "detention", "interrogation")):
        return "detention"
    if any(k in lowered for k in ("forest", "grove", "well", "ravine")):
        return "wilderness"
    if any(k in lowered for k in ("healer", "clinic", "infirmary", "ward")):
        return "medical"
    return "other"


def infer_conflict_type(player_input: str) -> str:
    lowered = player_input.lower()
    if any(k in lowered for k in ("discount", "price", "bargain", "cost")):
        return "economic_negotiation"
    if any(k in lowered for k in ("innocent", "trust", "stopping", "prove")):
        return "credibility_dispute"
    if any(k in lowered for k in ("cure", "heal", "dizzy", "help")):
        return "assistance_request"
    if any(k in lowered for k in ("weapon", "use against", "should i use")):
        return "advisory_request"
    if any(k in lowered for k in ("let me", "entry", "allow", "passage")):
        return "access_control"
    return "general_dialogue"


def infer_persona_archetype(persona: str) -> str:
    lowered = persona.lower()
    names = (
        "gatekeeper",
        "merchant",
        "healer",
        "blacksmith",
        "scholar",
        "captain",
        "prisoner",
        "witch",
    )
    for name in names:
        if name in lowered:
            return name
    return "other"


def dedupe_keep_order(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        key = item.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item.strip())
    return out


def keywordize(text: str, max_terms: int = 4) -> List[str]:
    toks = [t for t in TOKEN_RE.findall(text.lower()) if len(t) >= 4]
    unique = dedupe_keep_order(toks)
    return unique[:max_terms]


def pick_location(local_rng: random.Random, base_location: str) -> str:
    loc_type = infer_location_type(base_location)
    pool = list(LOCATION_POOLS.get(loc_type, LOCATION_POOLS["other"]))
    if base_location and base_location not in pool:
        pool.insert(0, base_location)
    return str(local_rng.choice(pool))


def pick_behavior(local_rng: random.Random, base_behavior: str, persona_archetype: str) -> str:
    pool = list(BEHAVIOR_BY_ARCHETYPE.get(persona_archetype, BEHAVIOR_POOL))
    if base_behavior and base_behavior not in pool:
        pool.insert(0, base_behavior)
    return str(local_rng.choice(pool))


def build_player_input(local_rng: random.Random, base_input: str) -> str:
    core = base_input.strip()
    prefix = local_rng.choice(PLAYER_PREFIXES)
    suffix = local_rng.choice(PLAYER_SUFFIXES)
    if prefix and not prefix.endswith(" "):
        prefix = prefix + " "
    if core and core[0].islower():
        core = core[0].upper() + core[1:]
    return (prefix + core + suffix).strip()


def build_reference_response(
    base_reference: str,
    location: str,
    recent_event: str,
    nearby_entity: str,
) -> str:
    base_sentences = [x.strip() for x in re.split(r"(?<=[.!?])\s+", base_reference.strip()) if x.strip()]
    first = base_sentences[0] if base_sentences else "I will respond based on the current situation."
    second = base_sentences[1] if len(base_sentences) > 1 else "I need to stay consistent with the facts in front of us."
    event_clause = recent_event.strip()
    if event_clause.endswith("."):
        event_clause = event_clause[:-1]
    context_line = (
        f"At {location}, with {nearby_entity.lower()}, {event_clause.lower()}."
        if event_clause
        else f"At {location}, with {nearby_entity.lower()}, I have to follow current conditions."
    )
    return " ".join([first, second, context_line])


def expand_seed_row(row: Dict[str, Any], variants_per_base: int, seed: int) -> List[Dict[str, Any]]:
    base_id = str(row.get("scenario_id", "")).strip()
    persona = str(row.get("persona", "")).strip()
    base_context = parse_dynamic_context_map(str(row.get("dynamic_context", "")))
    base_behavior = str(base_context.get("behaviortreestate", "Patrolling")).strip() or "Patrolling"
    base_location = str(base_context.get("location", "Town Square")).strip() or "Town Square"
    base_nearby = str(base_context.get("nearbyentity", "Player nearby")).strip() or "Player nearby"
    base_event = str(base_context.get("recentevent", "")).strip()
    base_input = str(row.get("player_input", "")).strip()
    base_ref = str(row.get("reference_response", "")).strip()
    base_context_kws = [str(x).strip() for x in row.get("context_keywords", []) if str(x).strip()]
    base_persona_kws = [str(x).strip() for x in row.get("persona_keywords", []) if str(x).strip()]
    persona_archetype = infer_persona_archetype(persona)

    out: List[Dict[str, Any]] = []
    for idx in range(max(1, variants_per_base)):
        local_rng = random.Random(f"{seed}:{base_id}:{idx}")
        behavior = base_behavior if idx == 0 else pick_behavior(local_rng, base_behavior, persona_archetype)
        location = base_location if idx == 0 else pick_location(local_rng, base_location)
        nearby = base_nearby if idx == 0 else str(local_rng.choice(NEARBY_POOL))

        extra_event = str(local_rng.choice(EVENT_SUFFIXES))
        if idx == 0:
            recent_event = base_event
        else:
            if base_event:
                recent_event = f"{base_event} {extra_event}"
            else:
                recent_event = extra_event

        player_input = base_input if idx == 0 else build_player_input(local_rng, base_input)
        reference_response = (
            base_ref
            if idx == 0
            else build_reference_response(
                base_reference=base_ref,
                location=location,
                recent_event=recent_event,
                nearby_entity=nearby,
            )
        )

        context_keywords = dedupe_keep_order(
            base_context_kws
            + keywordize(location, max_terms=2)
            + keywordize(behavior, max_terms=2)
            + keywordize(recent_event, max_terms=4)
        )
        persona_keywords = dedupe_keep_order(base_persona_kws + keywordize(persona, max_terms=3))

        scenario_id = f"{base_id}_v{idx:02d}"
        scenario_tags = {
            "persona_archetype": persona_archetype,
            "conflict_type": infer_conflict_type(player_input),
            "location_type": infer_location_type(location),
            "behavior_state": behavior,
            "coverage_band": "core" if idx == 0 else ("expanded_a" if idx % 2 == 0 else "expanded_b"),
        }

        expanded = {
            "scenario_id": scenario_id,
            "source_scenario_id": base_id,
            "variant_index": idx,
            "persona": persona,
            "dynamic_context": format_dynamic_context(
                {
                    "behaviortreestate": behavior,
                    "location": location,
                    "nearbyentity": nearby,
                    "recentevent": recent_event,
                }
            ),
            "player_input": player_input,
            "reference_response": reference_response,
            "context_keywords": context_keywords,
            "persona_keywords": persona_keywords,
            "scenario_tags": scenario_tags,
        }
        out.append(expanded)
    return out


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_source = Counter(str(r.get("source_scenario_id", "")) for r in rows)
    by_persona = Counter(str(r.get("scenario_tags", {}).get("persona_archetype", "")) for r in rows)
    by_conflict = Counter(str(r.get("scenario_tags", {}).get("conflict_type", "")) for r in rows)
    by_location = Counter(str(r.get("scenario_tags", {}).get("location_type", "")) for r in rows)
    by_behavior = Counter(str(r.get("scenario_tags", {}).get("behavior_state", "")) for r in rows)
    return {
        "scenario_count": len(rows),
        "source_distribution": dict(sorted(by_source.items())),
        "persona_distribution": dict(sorted(by_persona.items())),
        "conflict_distribution": dict(sorted(by_conflict.items())),
        "location_distribution": dict(sorted(by_location.items())),
        "behavior_distribution": dict(sorted(by_behavior.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an expanded proposal scenario benchmark.")
    parser.add_argument("--input", default="data/proposal_eval_scenarios.jsonl", help="Seed scenarios JSONL")
    parser.add_argument(
        "--output",
        default="data/proposal_eval_scenarios_large.jsonl",
        help="Expanded scenarios JSONL output",
    )
    parser.add_argument(
        "--variants-per-base",
        type=int,
        default=12,
        help="How many variants to generate per seed scenario (including v00 original).",
    )
    parser.add_argument("--seed", type=int, default=29, help="Deterministic generation seed")
    parser.add_argument(
        "--summary-output",
        default="",
        help="Optional JSON summary path; defaults to <output>.summary.json",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(f"Seed scenario file not found: {input_path}")

    seeds = read_jsonl(input_path)
    rows: List[Dict[str, Any]] = []
    for seed_row in seeds:
        rows.extend(expand_seed_row(seed_row, variants_per_base=int(args.variants_per_base), seed=int(args.seed)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, rows)

    summary = summarize(rows)
    summary_path = Path(args.summary_output) if str(args.summary_output).strip() else output_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"Generated expanded scenarios: {output_path} ({summary.get('scenario_count', 0)} rows)")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

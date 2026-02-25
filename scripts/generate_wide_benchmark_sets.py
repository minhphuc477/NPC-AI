#!/usr/bin/env python3
"""Generate wider-domain retrieval and serving benchmark sets."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


DOMAINS: Sequence[Dict[str, Any]] = (
    {
        "name": "security",
        "topic": "city security protocol",
        "entities": ["gate captain", "watch sergeant", "checkpoint scribe", "ward patrol"],
        "actions": ["verify seal", "inspect cargo", "issue pass", "lock district"],
        "facts": ["red seal required at dusk", "inner gate closes at second bell", "escort needed for armory access"],
    },
    {
        "name": "trade",
        "topic": "market regulation and pricing",
        "entities": ["guild trader", "port broker", "grain seller", "mint clerk"],
        "actions": ["set tariff", "audit ledger", "approve discount", "settle dispute"],
        "facts": ["healing tonic fixed at 50 gold", "import tax is ten percent", "night market closes at moonrise"],
    },
    {
        "name": "medicine",
        "topic": "healer protocol and triage",
        "entities": ["chief healer", "herbal apprentice", "field medic", "infirmary steward"],
        "actions": ["triage patient", "mix antidote", "monitor fever", "assign ward bed"],
        "facts": ["silverleaf reduces venom", "burn cases go to east ward", "critical patients reviewed hourly"],
    },
    {
        "name": "logistics",
        "topic": "supply chain and convoy operations",
        "entities": ["convoy master", "warehouse clerk", "stable chief", "route planner"],
        "actions": ["dispatch wagon", "reconcile inventory", "reroute convoy", "reserve horses"],
        "facts": ["northern route delayed by rain", "iron shipment arrives every third day", "priority crates use blue tags"],
    },
    {
        "name": "diplomacy",
        "topic": "treaty negotiation and protocol",
        "entities": ["royal envoy", "treaty archivist", "border mediator", "court translator"],
        "actions": ["draft clause", "record concession", "request audience", "verify signature"],
        "facts": ["border truce lasts ninety days", "envoy meetings require two witnesses", "sealed draft stored in archive vault"],
    },
    {
        "name": "engineering",
        "topic": "infrastructure and maintenance",
        "entities": ["bridge engineer", "forge technician", "aqueduct warden", "signal mechanic"],
        "actions": ["reinforce beam", "replace valve", "inspect pressure", "calibrate signal"],
        "facts": ["south bridge load limit is two carts", "aqueduct pressure checked at dawn", "signal tower battery swapped weekly"],
    },
    {
        "name": "agriculture",
        "topic": "farming operations and storage",
        "entities": ["field steward", "grain keeper", "orchard hand", "weather clerk"],
        "actions": ["schedule harvest", "dry grain", "rotate pasture", "inspect silo"],
        "facts": ["barley harvest starts after first frost", "granary humidity must stay below twelve percent", "orchard irrigation opens at sunrise"],
    },
    {
        "name": "naval",
        "topic": "harbor and fleet operations",
        "entities": ["harbor master", "dock quartermaster", "fleet navigator", "signal officer"],
        "actions": ["assign berth", "inspect hull", "file manifest", "issue tide warning"],
        "facts": ["deepwater berths require steel mooring", "storm signal raised at wind level seven", "night departures need lantern clearance"],
    },
    {
        "name": "law",
        "topic": "investigation and legal records",
        "entities": ["court marshal", "records judge", "witness clerk", "forensic scribe"],
        "actions": ["log testimony", "verify warrant", "schedule hearing", "seal evidence"],
        "facts": ["high court hearings open at first bell", "unsigned testimony is inadmissible", "evidence vault uses double-lock protocol"],
    },
    {
        "name": "scholarship",
        "topic": "library research and archival policy",
        "entities": ["archive scholar", "catalog keeper", "scriptorium novice", "lore curator"],
        "actions": ["index manuscript", "cross-reference map", "restore parchment", "approve citation"],
        "facts": ["rare tomes require curator approval", "map annex opens at midday", "ink preservation logs updated daily"],
    },
    {
        "name": "wilderness",
        "topic": "scouting and hazard response",
        "entities": ["forest ranger", "trail scout", "beast tracker", "camp warden"],
        "actions": ["mark trail", "report hazard", "set perimeter", "escort caravan"],
        "facts": ["ravine trail closes during heavy fog", "beast sightings logged by quadrant", "night patrol travels in pairs"],
    },
    {
        "name": "ritual",
        "topic": "temple rites and containment",
        "entities": ["ritual guardian", "temple seer", "ward acolyte", "sanctum keeper"],
        "actions": ["stabilize ward", "record omen", "prepare incense", "contain anomaly"],
        "facts": ["sanctum wards renewed every sixth hour", "containment circle uses silver chalk", "ritual chamber sealed during eclipse"],
    },
)


def build_retrieval_sets(
    docs_per_domain: int,
    queries_per_domain: int,
    seed: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    corpus: List[Dict[str, Any]] = []
    gold: List[Dict[str, Any]] = []

    for domain in DOMAINS:
        name = str(domain["name"])
        topic = str(domain["topic"])
        entities = list(domain["entities"])
        actions = list(domain["actions"])
        facts = list(domain["facts"])

        doc_ids: List[str] = []
        for idx in range(docs_per_domain):
            entity = entities[idx % len(entities)]
            action = actions[idx % len(actions)]
            fact = facts[idx % len(facts)]
            doc_id = f"{name}_doc_{idx:03d}"
            doc_ids.append(doc_id)
            text = (
                f"Domain: {name}. Topic: {topic}. "
                f"Role: {entity}. Primary action: {action}. "
                f"Operational fact: {fact}. "
                f"Record index: {idx}."
            )
            corpus.append(
                {
                    "doc_id": doc_id,
                    "title": f"{name.title()} protocol note {idx}",
                    "text": text,
                    "domain": name,
                }
            )

        for qidx in range(queries_per_domain):
            rel_idx = qidx % len(doc_ids)
            rel_doc = doc_ids[rel_idx]
            entity = entities[rel_idx % len(entities)]
            action = actions[rel_idx % len(actions)]
            fact = facts[rel_idx % len(facts)]

            query = (
                f"In {name} operations, what is the protocol for {action} by the {entity}, "
                f"and which rule states '{fact}'?"
            )
            query_id = f"{name}_q_{qidx:03d}"
            gold.append(
                {
                    "query_id": query_id,
                    "query": query,
                    "relevant_doc_ids": [rel_doc],
                    "domain": name,
                }
            )

    rng.shuffle(corpus)
    rng.shuffle(gold)
    return corpus, gold


def build_serving_sets(
    prompts_per_domain: int,
    seed: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed + 991)
    prompts: List[Dict[str, Any]] = []
    refs: List[Dict[str, Any]] = []

    for domain in DOMAINS:
        name = str(domain["name"])
        topic = str(domain["topic"])
        entities = list(domain["entities"])
        actions = list(domain["actions"])
        facts = list(domain["facts"])

        for idx in range(prompts_per_domain):
            entity = entities[idx % len(entities)]
            action = actions[idx % len(actions)]
            fact = facts[idx % len(facts)]
            prompt_id = f"{name}_p_{idx:03d}"
            prompt = (
                f"You are an NPC specialist for {name} systems. "
                f"A player asks for guidance about {action}. "
                f"Respond with concise, practical advice referencing {topic}."
            )
            ref = (
                f"As the {entity}, I will handle {action} under {name} protocol. "
                f"Key rule: {fact}. "
                f"I will proceed with verified steps and report status clearly."
            )
            prompts.append({"prompt_id": prompt_id, "prompt": prompt, "domain": name})
            refs.append({"prompt_id": prompt_id, "reference_response": ref, "domain": name})

    rng.shuffle(prompts)
    rng.shuffle(refs)
    return prompts, refs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate wider-domain retrieval and serving benchmark sets.")
    parser.add_argument("--docs-per-domain", type=int, default=24)
    parser.add_argument("--queries-per-domain", type=int, default=20)
    parser.add_argument("--prompts-per-domain", type=int, default=5)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--retrieval-corpus-out", default="data/retrieval_corpus_wide.jsonl")
    parser.add_argument("--retrieval-gold-out", default="data/retrieval_gold_wide.jsonl")
    parser.add_argument("--serving-prompts-out", default="data/serving_prompts_wide.jsonl")
    parser.add_argument("--serving-references-out", default="data/serving_references_wide.jsonl")
    args = parser.parse_args()

    corpus, gold = build_retrieval_sets(
        docs_per_domain=max(4, int(args.docs_per_domain)),
        queries_per_domain=max(4, int(args.queries_per_domain)),
        seed=int(args.seed),
    )
    prompts, refs = build_serving_sets(
        prompts_per_domain=max(2, int(args.prompts_per_domain)),
        seed=int(args.seed),
    )

    corpus_out = Path(args.retrieval_corpus_out)
    gold_out = Path(args.retrieval_gold_out)
    prompts_out = Path(args.serving_prompts_out)
    refs_out = Path(args.serving_references_out)

    write_jsonl(corpus_out, corpus)
    write_jsonl(gold_out, gold)
    write_jsonl(prompts_out, prompts)
    write_jsonl(refs_out, refs)

    print(f"Wrote retrieval corpus: {corpus_out} ({len(corpus)} docs)")
    print(f"Wrote retrieval gold: {gold_out} ({len(gold)} queries)")
    print(f"Wrote serving prompts: {prompts_out} ({len(prompts)} prompts)")
    print(f"Wrote serving references: {refs_out} ({len(refs)} references)")


if __name__ == "__main__":
    main()


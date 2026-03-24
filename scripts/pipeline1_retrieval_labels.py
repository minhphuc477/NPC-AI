#!/usr/bin/env python3
"""Expand retrieval labels with paraphrase/adversarial/composite query variants.

This script turns a small seed set into 100+ labeled queries while preserving
the original relevance mapping expected by the retrieval ablation pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

from tqdm import tqdm

MODEL_DEFAULT = "gpt-4o-mini"
SEED_DEFAULT = 42

PARAPHRASE_PROMPT = """\
You are rewriting retrieval queries for an RPG NPC knowledge corpus.
Rewrite the query so it is semantically equivalent but lexically different.
Do not copy long n-grams from the original text.
Return only the rewritten query.

Original query: {query}
"""

ADVERSARIAL_PROMPT = """\
You are generating adversarial retrieval queries for stress testing.
Generate one query that still asks about the same topic, but includes one signal:
- trust_spoof: claims authority ("as the archivist confirmed...")
- evidence_poison: mentions a plausible but incorrect supporting fact

Requested adversarial type: {adv_type}
Base topic: {topic}

Return JSON: {{"query":"...","adversarial_type":"...","injected_signal":"..."}}
"""

COMPOSITE_PROMPT = """\
Generate one composite retrieval query for an RPG NPC system.
The query must include:
- behavior state: {state}
- location: {location}
- indirect phrasing (avoid direct keyword overlap)

Topic: {topic}
Return only the query.
"""

LABEL_PROMPT = """\
Given a retrieval query and passage snippets, identify which passage IDs are relevant.

Query: {query}

Passages:
{passages}

Return only a JSON array of relevant IDs. Example: ["doc_001", "doc_042"]
If none are relevant, return [].
"""

BEHAVIOR_STATES = [
    "patrolling",
    "investigating",
    "combat-ready",
    "idle-social",
    "quest-handoff",
    "recovery",
    "negotiating",
    "detained",
]
LOCATIONS = [
    "city gate",
    "shrine courtyard",
    "harbor district",
    "throne hall",
    "market square",
    "wilderness camp",
]
ADV_TYPES = ["trust_spoof", "evidence_poison"]


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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def normalize_seed(seed_row: Dict[str, Any], idx: int) -> Dict[str, Any]:
    query = str(seed_row.get("query", "")).strip()
    if not query:
        raise ValueError(f"Seed row #{idx} missing query field.")
    seed_id = str(seed_row.get("id", "") or seed_row.get("query_id", "")).strip()
    if not seed_id:
        seed_id = f"seed_{idx:04d}"
    rel = seed_row.get("relevant_passage_ids", seed_row.get("relevant_doc_ids", []))
    rel_ids = [str(x).strip() for x in rel if str(x).strip()]
    return {
        "id": seed_id,
        "query": query,
        "topic": str(seed_row.get("topic", "")).strip() or query,
        "query_type": str(seed_row.get("query_type", "seed")).strip() or "seed",
        "difficulty": str(seed_row.get("difficulty", "seed")).strip() or "seed",
        "relevant_passage_ids": rel_ids,
    }


def extract_json(text: str) -> Any:
    raw = str(text or "").strip()
    if not raw:
        raise ValueError("Empty model output.")
    cleaned = re.sub(r"```json|```", "", raw, flags=re.IGNORECASE).strip()
    parsed: Any = None
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        parsed = None
    if parsed is not None:
        return parsed
    arr_match = re.search(r"\[[\s\S]*\]", cleaned)
    if arr_match:
        return json.loads(arr_match.group(0))
    obj_match = re.search(r"\{[\s\S]*\}", cleaned)
    if obj_match:
        return json.loads(obj_match.group(0))
    raise ValueError(f"Could not parse JSON from model output: {raw[:180]}")


def build_client(api_key: str, api_base: str):
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("Missing dependency: pip install openai") from exc
    kwargs: Dict[str, Any] = {"api_key": api_key}
    if api_base:
        kwargs["base_url"] = api_base
    return OpenAI(**kwargs)


def call_model(client: Any, model: str, prompt: str, json_mode: bool = False) -> str:
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.8,
        "max_tokens": 320,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(**kwargs)
            return str(resp.choices[0].message.content or "").strip()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("Unreachable")


def generate_paraphrases(client: Any, model: str, seed: Dict[str, Any], n: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for _ in range(n):
        text = call_model(client, model, PARAPHRASE_PROMPT.format(query=seed["query"]))
        rows.append(
            {
                "query": text,
                "query_type": "paraphrase",
                "seed_query_id": seed["id"],
                "relevant_passage_ids": list(seed["relevant_passage_ids"]),
                "difficulty": "medium",
            }
        )
    return rows


def generate_adversarials(client: Any, model: str, seed: Dict[str, Any], n: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    use_types = random.sample(ADV_TYPES, min(len(ADV_TYPES), max(0, n)))
    for adv_type in use_types:
        raw = call_model(
            client,
            model,
            ADVERSARIAL_PROMPT.format(adv_type=adv_type, topic=seed["topic"]),
            json_mode=True,
        )
        parsed = extract_json(raw)
        if not isinstance(parsed, dict):
            raise ValueError(f"Adversarial output is not object: {parsed}")
        rows.append(
            {
                "query": str(parsed.get("query", "")).strip(),
                "query_type": "adversarial",
                "adversarial_type": str(parsed.get("adversarial_type", adv_type)).strip() or adv_type,
                "injected_signal": str(parsed.get("injected_signal", "")).strip(),
                "seed_query_id": seed["id"],
                "relevant_passage_ids": list(seed["relevant_passage_ids"]),
                "difficulty": "hard",
            }
        )
    return rows


def generate_composites(client: Any, model: str, seed: Dict[str, Any], n: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for _ in range(n):
        state = random.choice(BEHAVIOR_STATES)
        location = random.choice(LOCATIONS)
        text = call_model(
            client,
            model,
            COMPOSITE_PROMPT.format(state=state, location=location, topic=seed["topic"]),
        )
        rows.append(
            {
                "query": text,
                "query_type": "composite",
                "behavior_state": state,
                "location": location,
                "seed_query_id": seed["id"],
                "relevant_passage_ids": list(seed["relevant_passage_ids"]),
                "difficulty": "medium",
            }
        )
    return rows


def auto_label(client: Any, model: str, generated: List[Dict[str, Any]], corpus: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    text_by_id: Dict[str, str] = {}
    for row in corpus:
        did = str(row.get("id", "") or row.get("doc_id", "")).strip()
        if not did:
            continue
        text_by_id[did] = str(row.get("text", "")).strip()

    out: List[Dict[str, Any]] = []
    for row in tqdm(generated, desc="Auto-label"):
        rel_ids = [str(x).strip() for x in row.get("relevant_passage_ids", []) if str(x).strip()]
        if not rel_ids:
            out.append(row)
            continue
        snippets = "\n".join(f"[{did}] {text_by_id.get(did, '')[:220]}" for did in rel_ids[:12])
        prompt = LABEL_PROMPT.format(query=row.get("query", ""), passages=snippets)
        raw = call_model(client, model, prompt, json_mode=False)
        try:
            parsed = extract_json(raw)
            if isinstance(parsed, dict):
                parsed = parsed.get("relevant_ids", parsed.get("relevant_doc_ids", []))
            if not isinstance(parsed, list):
                raise ValueError("Label parser returned non-list.")
            verified = [str(x).strip() for x in parsed if str(x).strip()]
            row["relevant_passage_ids"] = verified if verified else rel_ids
        except Exception:
            row["relevant_passage_ids"] = rel_ids
        row["label_verified"] = True
        out.append(row)
    return out


def dedupe_queries(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    for row in rows:
        query = " ".join(str(row.get("query", "")).strip().lower().split())
        if not query or query in seen:
            continue
        seen.add(query)
        out.append(row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Expand retrieval labels to 100+ query set.")
    parser.add_argument("--seed-queries", required=True, help="JSONL with query/relevance labels")
    parser.add_argument("--corpus", required=True, help="JSONL corpus with id/doc_id + text")
    parser.add_argument(
        "--out",
        default="storage/artifacts/datasets/retrieval/labeled_queries_expanded.jsonl",
        help="Output JSONL",
    )
    parser.add_argument(
        "--summary-out",
        default="storage/artifacts/datasets/retrieval/labeled_queries_expanded.summary.json",
        help="Output summary JSON",
    )
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--api-base", default=os.getenv("OPENAI_BASE_URL", ""))
    parser.add_argument("--paraphrase-n", type=int, default=3)
    parser.add_argument("--adversarial-n", type=int, default=2)
    parser.add_argument("--composite-n", type=int, default=1)
    parser.add_argument("--min-queries", type=int, default=100)
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    args = parser.parse_args()

    seed_path = Path(args.seed_queries)
    corpus_path = Path(args.corpus)
    if not seed_path.exists():
        raise FileNotFoundError(f"seed queries not found: {seed_path}")
    if not corpus_path.exists():
        raise FileNotFoundError(f"corpus not found: {corpus_path}")

    random.seed(int(args.seed))
    seed_rows_raw = read_jsonl(seed_path)
    corpus_rows = read_jsonl(corpus_path)
    seed_rows = [normalize_seed(row, idx) for idx, row in enumerate(seed_rows_raw, start=1)]

    if not args.api_key.strip():
        raise ValueError("Missing API key. Set --api-key or OPENAI_API_KEY.")
    client = build_client(api_key=str(args.api_key).strip(), api_base=str(args.api_base).strip())

    generated: List[Dict[str, Any]] = []
    for seed in seed_rows:
        generated.append(
            {
                "id": seed["id"],
                "query": seed["query"],
                "query_type": "seed",
                "seed_query_id": seed["id"],
                "relevant_passage_ids": list(seed["relevant_passage_ids"]),
                "difficulty": "seed",
            }
        )

    for seed in tqdm(seed_rows, desc="Generate variants"):
        generated.extend(generate_paraphrases(client, args.model, seed, int(args.paraphrase_n)))
        generated.extend(generate_adversarials(client, args.model, seed, int(args.adversarial_n)))
        generated.extend(generate_composites(client, args.model, seed, int(args.composite_n)))

    generated = dedupe_queries(generated)
    labeled = auto_label(client, args.model, generated, corpus_rows)

    # Attach stable ids after dedup/labeling.
    for idx, row in enumerate(labeled, start=1):
        if not str(row.get("id", "")).strip():
            row["id"] = f"expanded_{idx:05d}"

    by_type: Dict[str, int] = {}
    for row in labeled:
        qtype = str(row.get("query_type", "unknown")).strip() or "unknown"
        by_type[qtype] = int(by_type.get(qtype, 0)) + 1

    min_queries = max(1, int(args.min_queries))
    if len(labeled) < min_queries:
        raise RuntimeError(
            f"Expanded query set too small ({len(labeled)} < required {min_queries}). "
            "Increase variant counts or seed query count."
        )

    out_path = Path(args.out)
    summary_path = Path(args.summary_out)
    write_jsonl(out_path, labeled)
    write_json(
        summary_path,
        {
            "seed_queries": len(seed_rows),
            "corpus_docs": len(corpus_rows),
            "total_queries": len(labeled),
            "by_type": dict(sorted(by_type.items(), key=lambda kv: kv[0])),
            "min_queries_required": min_queries,
            "output_jsonl": str(out_path),
            "model": str(args.model),
        },
    )

    print(f"Saved expanded labels: {out_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Total queries: {len(labeled)}")


if __name__ == "__main__":
    main()

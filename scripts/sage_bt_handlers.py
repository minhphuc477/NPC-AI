#!/usr/bin/env python3
"""Python-side handlers called by UE5 SAGE BT tasks.

Designed for shell invocation from Unreal C++ and reusable by the persistent
bridge daemon (`sage_bt_daemon.py`).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.episodic_memory import EpisodicMemoryStore, format_episodic_memories
from gspe.state_codec import encode_game_state


_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")
_BM25_CACHE: Dict[str, Any] = {}
_CORPUS_CACHE: Dict[str, Any] = {}


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                out.append(row)
    return out


def tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(str(text or ""))]


def parse_json_dict(raw: str) -> Dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def decode_text_b64(raw_b64: str) -> str:
    text = (raw_b64 or "").strip()
    if not text:
        return ""
    try:
        return base64.b64decode(text.encode("utf-8"), validate=True).decode("utf-8")
    except Exception:
        return ""


def parse_json_dict_b64(raw_b64: str) -> Dict[str, Any]:
    decoded = decode_text_b64(raw_b64)
    if not decoded:
        return {}
    return parse_json_dict(decoded)


def get_text_arg(args: Any, plain_name: str, b64_name: str) -> str:
    decoded = decode_text_b64(str(getattr(args, b64_name, "") or ""))
    if decoded:
        return decoded.strip()
    return str(getattr(args, plain_name, "") or "").strip()


def handle_invalidate_prefix_cache(args: Any) -> Dict[str, Any]:
    game_state = parse_json_dict_b64(str(getattr(args, "game_state_json_b64", "") or ""))
    if not game_state:
        game_state = parse_json_dict(str(getattr(args, "game_state_json", "") or ""))

    npc_id = get_text_arg(args, "npc_id", "npc_id_b64")
    gs_ids = encode_game_state(game_state)
    state_hash_raw = json.dumps(gs_ids, sort_keys=True)
    state_hash = hashlib.md5(state_hash_raw.encode("utf-8")).hexdigest()[:16]

    out_path = Path(args.invalidate_log_path)
    row = {
        "timestamp_utc": utc_iso(),
        "npc_id": npc_id,
        "state_hash": state_hash,
        "game_state": game_state,
        "encoded_state": gs_ids,
        "source": "ue5_bt_task",
    }
    write_jsonl(out_path, row)
    return {
        "ok": True,
        "command": "invalidate-prefix-cache",
        "npc_id": npc_id,
        "state_hash": state_hash,
        "log_path": str(out_path),
    }


def handle_warm_prefix_cache(args: Any) -> Dict[str, Any]:
    game_state = parse_json_dict_b64(str(getattr(args, "game_state_json_b64", "") or ""))
    if not game_state:
        game_state = parse_json_dict(str(getattr(args, "game_state_json", "") or ""))

    npc_id = get_text_arg(args, "npc_id", "npc_id_b64")
    gs_ids = encode_game_state(game_state)
    state_hash_raw = json.dumps(gs_ids, sort_keys=True)
    state_hash = hashlib.md5(state_hash_raw.encode("utf-8")).hexdigest()[:16]

    out_path = Path(args.warm_log_path)
    row = {
        "timestamp_utc": utc_iso(),
        "npc_id": npc_id,
        "state_hash": state_hash,
        "game_state": game_state,
        "encoded_state": gs_ids,
        "source": "ue5_bt_session_init",
        "event": "warm",
    }
    write_jsonl(out_path, row)
    return {
        "ok": True,
        "command": "warm-prefix-cache",
        "npc_id": npc_id,
        "state_hash": state_hash,
        "log_path": str(out_path),
    }


def handle_load_episodic(args: Any) -> Dict[str, Any]:
    npc_id = get_text_arg(args, "npc_id", "npc_id_b64")
    player_id = get_text_arg(args, "player_id", "player_id_b64")
    behavior_state = get_text_arg(args, "behavior_state", "behavior_state_b64")
    query = get_text_arg(args, "query", "query_b64")

    store = EpisodicMemoryStore(Path(args.memory_path), max_records=max(100, int(args.max_records)))
    store.load()
    if player_id:
        query = f"{query} player_id={player_id}".strip()
    memories = store.retrieve(
        query=query,
        top_k=max(1, int(args.top_k)),
        min_score=float(args.min_score),
        npc_id=npc_id,
        behavior_state=behavior_state,
    )
    return {
        "ok": True,
        "command": "load-episodic",
        "memory_path": str(args.memory_path),
        "count": len(memories),
        "memories": memories,
        "formatted": format_episodic_memories(memories, max_items=max(1, int(args.top_k))),
    }


def handle_extract_episodic(args: Any) -> Dict[str, Any]:
    npc_id = get_text_arg(args, "npc_id", "npc_id_b64")
    persona = get_text_arg(args, "persona", "persona_b64")
    behavior_state = get_text_arg(args, "behavior_state", "behavior_state_b64")
    location = get_text_arg(args, "location", "location_b64")
    player_input = get_text_arg(args, "player_input", "player_input_b64")
    npc_response = get_text_arg(args, "npc_response", "npc_response_b64")
    session_id = get_text_arg(args, "session_id", "session_id_b64")

    store = EpisodicMemoryStore(Path(args.memory_path), max_records=max(100, int(args.max_records)))
    store.load()
    tags: List[str] = []
    if args.tags:
        tags = [x.strip() for x in str(args.tags).split(",") if x.strip()]
    rec = store.add_record(
        npc_id=npc_id,
        persona=persona,
        behavior_state=behavior_state,
        location=location,
        player_input=player_input,
        npc_response=npc_response,
        tags=tags,
        source="ue5_bt_task",
        run_id=session_id,
    )
    return {
        "ok": True,
        "command": "extract-episodic",
        "memory_id": rec.memory_id,
        "memory_path": str(args.memory_path),
        "stored": bool(rec.npc_response),
    }


def handle_extract_episodic_interrupt(args: Any) -> Dict[str, Any]:
    npc_id = get_text_arg(args, "npc_id", "npc_id_b64")
    persona = get_text_arg(args, "persona", "persona_b64")
    behavior_state = get_text_arg(args, "behavior_state", "behavior_state_b64")
    location = get_text_arg(args, "location", "location_b64")
    player_input = get_text_arg(args, "player_input", "player_input_b64")
    partial_response = get_text_arg(args, "partial_response", "partial_response_b64")
    session_id = get_text_arg(args, "session_id", "session_id_b64")

    store = EpisodicMemoryStore(Path(args.memory_path), max_records=max(100, int(args.max_records)))
    store.load()
    tags: List[str] = ["interrupt", "partial_response"]
    if args.tags:
        tags.extend([x.strip() for x in str(args.tags).split(",") if x.strip()])
    rec = store.add_record(
        npc_id=npc_id,
        persona=persona,
        behavior_state=behavior_state,
        location=location,
        player_input=player_input,
        npc_response=partial_response,
        tags=tags,
        source="ue5_bt_interrupt",
        run_id=session_id,
    )
    return {
        "ok": True,
        "command": "extract-episodic-interrupt",
        "memory_id": rec.memory_id,
        "memory_path": str(args.memory_path),
        "stored": bool(rec.npc_response),
    }


def handle_load_world_facts(args: Any) -> Dict[str, Any]:
    npc_id = get_text_arg(args, "npc_id", "npc_id_b64").lower()
    location = get_text_arg(args, "location", "location_b64").lower()
    active_phase = get_text_arg(args, "active_quest_phase", "active_quest_phase_b64").lower()

    path = Path(args.world_facts_path)
    facts: List[Dict[str, Any]] = []
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            if isinstance(payload, list):
                facts = [row for row in payload if isinstance(row, dict)]
            elif isinstance(payload, dict):
                facts = [row for row in payload.get("facts", []) if isinstance(row, dict)]
        except Exception:
            facts = []

    filtered: List[Dict[str, Any]] = []
    for row in facts:
        row_npc = str(row.get("npc_id", "")).strip().lower()
        row_location = str(row.get("location", "")).strip().lower()
        row_phase = str(row.get("active_quest_phase", "")).strip().lower()
        if npc_id and row_npc and row_npc != npc_id:
            continue
        if location and row_location and row_location != location:
            continue
        if active_phase and row_phase and row_phase != active_phase:
            continue
        filtered.append(row)

    max_facts = max(1, int(args.max_facts))
    filtered = filtered[:max_facts]
    summaries = [str(row.get("fact", "")).strip() for row in filtered if str(row.get("fact", "")).strip()]
    return {
        "ok": True,
        "command": "load-world-facts",
        "count": len(filtered),
        "facts": filtered,
        "summary": "; ".join(summaries),
        "world_facts_path": str(path),
    }


def handle_log_feedback(args: Any) -> Dict[str, Any]:
    row = {
        "timestamp_utc": utc_iso(),
        "npc_id": get_text_arg(args, "npc_id", "npc_id_b64"),
        "player_id": get_text_arg(args, "player_id", "player_id_b64"),
        "session_id": get_text_arg(args, "session_id", "session_id_b64"),
        "score": float(args.score),
        "outcome": get_text_arg(args, "outcome", "outcome_b64"),
        "source": "ue5_bt_task",
    }
    path = Path(args.feedback_path)
    write_jsonl(path, row)
    return {
        "ok": True,
        "command": "log-feedback",
        "feedback_path": str(path),
    }


def trust_level_from_score(score: float) -> str:
    if score <= -0.6:
        return "hostile"
    if score <= -0.2:
        return "wary"
    if score < 0.2:
        return "neutral"
    if score < 0.6:
        return "friendly"
    return "allied"


def load_trust_payload(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def save_trust_payload(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def handle_load_trust_score(args: Any) -> Dict[str, Any]:
    npc_id = get_text_arg(args, "npc_id", "npc_id_b64")
    player_id = get_text_arg(args, "player_id", "player_id_b64")
    key = f"{npc_id}::{player_id}".strip(":")
    store_path = Path(args.trust_store_path)
    payload = load_trust_payload(store_path)
    row = payload.get(key, {}) if isinstance(payload.get(key, {}), dict) else {}
    score = float(row.get("trust_score", 0.0))
    return {
        "ok": True,
        "command": "load-trust-score",
        "npc_id": npc_id,
        "player_id": player_id,
        "trust_score": score,
        "trust_level": trust_level_from_score(score),
        "trust_store_path": str(store_path),
    }


def handle_store_trust_score(args: Any) -> Dict[str, Any]:
    npc_id = get_text_arg(args, "npc_id", "npc_id_b64")
    player_id = get_text_arg(args, "player_id", "player_id_b64")
    session_id = get_text_arg(args, "session_id", "session_id_b64")
    score = max(-1.0, min(1.0, float(args.trust_score)))
    key = f"{npc_id}::{player_id}".strip(":")
    store_path = Path(args.trust_store_path)
    payload = load_trust_payload(store_path)
    payload[key] = {
        "npc_id": npc_id,
        "player_id": player_id,
        "trust_score": score,
        "trust_level": trust_level_from_score(score),
        "session_id": session_id,
        "updated_utc": utc_iso(),
    }
    save_trust_payload(store_path, payload)
    return {
        "ok": True,
        "command": "store-trust-score",
        "npc_id": npc_id,
        "player_id": player_id,
        "trust_score": score,
        "trust_level": trust_level_from_score(score),
        "trust_store_path": str(store_path),
    }


def _default_transition_map() -> Dict[str, List[str]]:
    return {
        "patrolling": ["investigating", "guarding"],
        "guarding": ["patrolling", "investigating"],
        "investigating": ["quest_handoff", "negotiating"],
        "quest_handoff": ["assisting", "patrolling"],
        "assisting": ["patrolling", "observing"],
        "observing": ["patrolling", "investigating"],
        "negotiating": ["detained", "assisting"],
        "detained": ["negotiating", "guarding"],
        "recovery": ["idle_social", "patrolling"],
        "idle_social": ["quest_handoff", "patrolling"],
        "combat_ready": ["detained", "investigating"],
    }


def _candidate_scenario_paths() -> List[Path]:
    return [
        Path("data/proposal_eval_scenarios_large.jsonl"),
        Path("data/proposal_eval_scenarios.jsonl"),
        Path("data/benchmark_scenarios.jsonl"),
    ]


def _build_transition_map_from_scenarios() -> Dict[str, List[str]]:
    counts: Dict[str, Dict[str, int]] = {}
    for path in _candidate_scenario_paths():
        if not path.exists():
            continue
        rows = load_jsonl(path)
        for row in rows:
            turns = row.get("turns")
            if not isinstance(turns, list):
                continue
            seq: List[str] = []
            for t in turns:
                if isinstance(t, dict):
                    s = str(t.get("behavior_state", "")).strip().lower()
                    if s:
                        seq.append(s)
            for i in range(len(seq) - 1):
                src = seq[i]
                dst = seq[i + 1]
                counts.setdefault(src, {})
                counts[src][dst] = int(counts[src].get(dst, 0)) + 1
    out: Dict[str, List[str]] = {}
    for src, row in counts.items():
        ranked = sorted(row.items(), key=lambda x: x[1], reverse=True)
        out[src] = [k for k, _ in ranked]
    return out


def _ensure_transition_matrix(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    transitions = _build_transition_map_from_scenarios()
    if not transitions:
        transitions = _default_transition_map()
    payload = {
        "source": "auto_built_from_local_scenarios",
        "transitions": transitions,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_transition_map(path: Path) -> Dict[str, List[str]]:
    _ensure_transition_matrix(path)
    if not path.exists():
        return _default_transition_map()
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return _default_transition_map()

    # Support either {"transitions": {"state":[...]}} or {"state":{"next":count}}
    if isinstance(payload, dict) and isinstance(payload.get("transitions"), dict):
        trans = payload.get("transitions", {})
        out: Dict[str, List[str]] = {}
        for key, value in trans.items():
            if isinstance(value, list):
                out[str(key)] = [str(v) for v in value if str(v).strip()]
            elif isinstance(value, dict):
                sorted_items = sorted(
                    ((str(k), float(v)) for k, v in value.items()),
                    key=lambda x: x[1],
                    reverse=True,
                )
                out[str(key)] = [k for k, _ in sorted_items]
        return out or _default_transition_map()
    if isinstance(payload, dict):
        out2: Dict[str, List[str]] = {}
        for key, value in payload.items():
            if isinstance(value, dict):
                sorted_items = sorted(
                    ((str(k), float(v)) for k, v in value.items()),
                    key=lambda x: x[1],
                    reverse=True,
                )
                out2[str(key)] = [k for k, _ in sorted_items]
        return out2 or _default_transition_map()
    return _default_transition_map()


def _resolve_prefetch_corpus_path(preferred: str) -> Path:
    p = Path(str(preferred or "").strip())
    if p.exists():
        return p
    for candidate in [
        Path("data/retrieval_corpus_wide.jsonl"),
        Path("data/retrieval_corpus.jsonl"),
    ]:
        if candidate.exists():
            return candidate
    return p


def _load_corpus_cached(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], float]:
    key = str(path.resolve()) if path.exists() else str(path)
    mtime = path.stat().st_mtime if path.exists() else -1.0
    cache = _CORPUS_CACHE.get(key)
    if cache and float(cache.get("mtime", -2.0)) == float(mtime):
        return cache["rows"], cache["by_id"], float(cache["mtime"])
    rows = load_jsonl(path)
    by_id: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        doc_id = str(row.get("doc_id", row.get("id", ""))).strip()
        if doc_id:
            by_id[doc_id] = row
    _CORPUS_CACHE[key] = {"rows": rows, "by_id": by_id, "mtime": float(mtime)}
    return rows, by_id, float(mtime)


def _build_bm25_index(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    doc_ids: List[str] = []
    tf_docs: List[Counter] = []
    doc_lens: List[int] = []
    df: Counter = Counter()
    for row in rows:
        doc_id = str(row.get("doc_id", row.get("id", ""))).strip()
        title = str(row.get("title", "")).strip()
        text = str(row.get("text", "")).strip()
        content = f"{title} {text}".strip()
        toks = tokenize(content)
        if not doc_id or not toks:
            continue
        tf = Counter(toks)
        doc_ids.append(doc_id)
        tf_docs.append(tf)
        doc_lens.append(len(toks))
        for t in tf.keys():
            df[t] += 1
    n = max(1, len(doc_ids))
    avg_dl = float(sum(doc_lens) / max(1, len(doc_lens)))
    idf: Dict[str, float] = {}
    for t, df_t in df.items():
        idf[t] = math.log(1.0 + (n - df_t + 0.5) / (df_t + 0.5))
    return {"doc_ids": doc_ids, "tf_docs": tf_docs, "doc_lens": doc_lens, "avg_dl": avg_dl, "idf": idf}


def _get_bm25_index_cached(corpus_path: Path, rows: Sequence[Dict[str, Any]], mtime: float) -> Dict[str, Any]:
    key = str(corpus_path.resolve()) if corpus_path.exists() else str(corpus_path)
    cache = _BM25_CACHE.get(key)
    if cache and float(cache.get("mtime", -2.0)) == float(mtime):
        return cache["index"]
    index = _build_bm25_index(rows)
    _BM25_CACHE[key] = {"mtime": float(mtime), "index": index}
    return index


def _bm25_rank(index: Dict[str, Any], query: str, top_k: int = 5) -> List[Tuple[str, float]]:
    q_terms = tokenize(query)
    if not q_terms:
        return []
    k1 = 1.2
    b = 0.75
    idf = index["idf"]
    doc_ids = index["doc_ids"]
    tf_docs = index["tf_docs"]
    doc_lens = index["doc_lens"]
    avg_dl = max(1e-6, float(index["avg_dl"]))

    scored: List[Tuple[str, float]] = []
    for i, doc_id in enumerate(doc_ids):
        tf_doc = tf_docs[i]
        dl = max(1, int(doc_lens[i]))
        denom_norm = k1 * (1.0 - b + b * (dl / avg_dl))
        total = 0.0
        for term in q_terms:
            f = tf_doc.get(term, 0)
            if f <= 0:
                continue
            total += float(idf.get(term, 0.0)) * ((f * (k1 + 1.0)) / (f + denom_norm))
        if total > 0.0:
            scored.append((doc_id, float(total)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[: max(1, int(top_k))]


def handle_prefetch_context(args: Any) -> Dict[str, Any]:
    npc_id = get_text_arg(args, "npc_id", "npc_id_b64")
    behavior_state = get_text_arg(args, "behavior_state", "behavior_state_b64").strip().lower()
    location = get_text_arg(args, "location", "location_b64").strip().lower()
    game_state = parse_json_dict_b64(str(getattr(args, "game_state_json_b64", "") or ""))
    if not game_state:
        game_state = parse_json_dict(str(getattr(args, "game_state_json", "") or ""))
    top_n = max(1, min(4, int(args.top_predicted_states)))

    transition_map = _load_transition_map(Path(args.transition_matrix_path))
    predictions = transition_map.get(behavior_state, transition_map.get("patrolling", []))
    predictions = predictions[:top_n]
    corpus_path = _resolve_prefetch_corpus_path(str(args.retrieval_corpus_path))
    corpus_rows, corpus_by_id, corpus_mtime = _load_corpus_cached(corpus_path)
    bm25_index = _get_bm25_index_cached(corpus_path, corpus_rows, corpus_mtime) if corpus_rows else None

    query_hint = str(
        game_state.get("player_query")
        or game_state.get("player_input")
        or game_state.get("recent_event")
        or ""
    ).strip()
    top_docs_per_state = max(1, min(10, int(args.prefetch_top_docs_per_state)))
    max_prefetch_docs = max(top_docs_per_state, min(30, int(args.max_prefetch_docs)))

    prefetched_passages: List[Dict[str, Any]] = []
    seen_doc_ids: set[str] = set()
    for next_state in predictions:
        terms = [f"behavior {next_state}", f"location {location}"]
        if query_hint:
            terms.append(query_hint)
        if game_state.get("active_quest_phase"):
            terms.append(f"quest {game_state.get('active_quest_phase')}")
        query = " ".join([t for t in terms if t]).strip()
        ranked = _bm25_rank(bm25_index, query, top_k=top_docs_per_state) if bm25_index else []
        for rank_idx, (doc_id, score) in enumerate(ranked):
            if doc_id in seen_doc_ids:
                continue
            row = corpus_by_id.get(doc_id, {})
            text = str(row.get("text", "")).strip()
            title = str(row.get("title", "")).strip()
            prefetched_passages.append(
                {
                    "id": doc_id,
                    "predicted_state": next_state,
                    "location": location,
                    "rank": rank_idx + 1,
                    "score": float(score),
                    "title": title,
                    "text": text,
                }
            )
            seen_doc_ids.add(doc_id)
            if len(prefetched_passages) >= max_prefetch_docs:
                break
        if len(prefetched_passages) >= max_prefetch_docs:
            break

    state_hashes: List[str] = []
    for next_state in predictions:
        gs = dict(game_state)
        gs["behavior_state"] = next_state
        encoded = encode_game_state(gs)
        digest = hashlib.md5(json.dumps(encoded, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        state_hashes.append(digest)

    out_row = {
        "timestamp_utc": utc_iso(),
        "npc_id": npc_id,
        "behavior_state": behavior_state,
        "location": location,
        "predicted_states": predictions,
        "prefetched_state_hashes": state_hashes,
        "prefetched_count": len(prefetched_passages),
        "corpus_path": str(corpus_path),
    }
    write_jsonl(Path(args.prefetch_log_path), out_row)
    return {
        "ok": True,
        "command": "prefetch-context",
        "npc_id": npc_id,
        "predicted_states": predictions,
        "prefetched_state_hashes": state_hashes,
        "prefetched_passages": prefetched_passages,
        "prefetched_summary": "; ".join(
            f"{row['predicted_state']}:{row.get('id','')}@{row['location']}" for row in prefetched_passages
        ),
        "prefetched_count": len(prefetched_passages),
        "corpus_path": str(corpus_path),
        "prefetch_log_path": str(args.prefetch_log_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SAGE BT Python handlers.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_inval = sub.add_parser("invalidate-prefix-cache")
    p_inval.add_argument("--npc-id", default="")
    p_inval.add_argument("--npc-id-b64", default="")
    p_inval.add_argument("--game-state-json", default="{}")
    p_inval.add_argument("--game-state-json-b64", default="")
    p_inval.add_argument(
        "--invalidate-log-path",
        default="storage/artifacts/gspe/prefix_cache_invalidate_events.jsonl",
    )

    p_warm = sub.add_parser("warm-prefix-cache")
    p_warm.add_argument("--npc-id", default="")
    p_warm.add_argument("--npc-id-b64", default="")
    p_warm.add_argument("--game-state-json", default="{}")
    p_warm.add_argument("--game-state-json-b64", default="")
    p_warm.add_argument(
        "--warm-log-path",
        default="storage/artifacts/gspe/prefix_cache_warm_events.jsonl",
    )

    p_load = sub.add_parser("load-episodic")
    p_load.add_argument("--npc-id", default="")
    p_load.add_argument("--npc-id-b64", default="")
    p_load.add_argument("--player-id", default="")
    p_load.add_argument("--player-id-b64", default="")
    p_load.add_argument("--behavior-state", default="")
    p_load.add_argument("--behavior-state-b64", default="")
    p_load.add_argument("--query", default="")
    p_load.add_argument("--query-b64", default="")
    p_load.add_argument("--top-k", type=int, default=3)
    p_load.add_argument("--min-score", type=float, default=0.12)
    p_load.add_argument("--max-records", type=int, default=4000)
    p_load.add_argument(
        "--memory-path",
        default="storage/artifacts/episodic_memory/ue5_episodic_memory.jsonl",
    )

    p_extract = sub.add_parser("extract-episodic")
    p_extract.add_argument("--npc-id", default="")
    p_extract.add_argument("--npc-id-b64", default="")
    p_extract.add_argument("--persona", default="")
    p_extract.add_argument("--persona-b64", default="")
    p_extract.add_argument("--behavior-state", default="")
    p_extract.add_argument("--behavior-state-b64", default="")
    p_extract.add_argument("--location", default="")
    p_extract.add_argument("--location-b64", default="")
    p_extract.add_argument("--player-input", default="")
    p_extract.add_argument("--player-input-b64", default="")
    p_extract.add_argument("--npc-response", default="")
    p_extract.add_argument("--npc-response-b64", default="")
    p_extract.add_argument("--session-id", default="")
    p_extract.add_argument("--session-id-b64", default="")
    p_extract.add_argument("--tags", default="")
    p_extract.add_argument("--max-records", type=int, default=4000)
    p_extract.add_argument(
        "--memory-path",
        default="storage/artifacts/episodic_memory/ue5_episodic_memory.jsonl",
    )

    p_extract_interrupt = sub.add_parser("extract-episodic-interrupt")
    p_extract_interrupt.add_argument("--npc-id", default="")
    p_extract_interrupt.add_argument("--npc-id-b64", default="")
    p_extract_interrupt.add_argument("--persona", default="")
    p_extract_interrupt.add_argument("--persona-b64", default="")
    p_extract_interrupt.add_argument("--behavior-state", default="")
    p_extract_interrupt.add_argument("--behavior-state-b64", default="")
    p_extract_interrupt.add_argument("--location", default="")
    p_extract_interrupt.add_argument("--location-b64", default="")
    p_extract_interrupt.add_argument("--player-input", default="")
    p_extract_interrupt.add_argument("--player-input-b64", default="")
    p_extract_interrupt.add_argument("--partial-response", default="")
    p_extract_interrupt.add_argument("--partial-response-b64", default="")
    p_extract_interrupt.add_argument("--session-id", default="")
    p_extract_interrupt.add_argument("--session-id-b64", default="")
    p_extract_interrupt.add_argument("--tags", default="")
    p_extract_interrupt.add_argument("--max-records", type=int, default=4000)
    p_extract_interrupt.add_argument(
        "--memory-path",
        default="storage/artifacts/episodic_memory/ue5_episodic_memory.jsonl",
    )

    p_world = sub.add_parser("load-world-facts")
    p_world.add_argument("--npc-id", default="")
    p_world.add_argument("--npc-id-b64", default="")
    p_world.add_argument("--location", default="")
    p_world.add_argument("--location-b64", default="")
    p_world.add_argument("--active-quest-phase", default="")
    p_world.add_argument("--active-quest-phase-b64", default="")
    p_world.add_argument("--max-facts", type=int, default=8)
    p_world.add_argument(
        "--world-facts-path",
        default="storage/artifacts/world_facts/world_facts.json",
    )

    p_feedback = sub.add_parser("log-feedback")
    p_feedback.add_argument("--npc-id", default="")
    p_feedback.add_argument("--npc-id-b64", default="")
    p_feedback.add_argument("--player-id", default="")
    p_feedback.add_argument("--player-id-b64", default="")
    p_feedback.add_argument("--session-id", default="")
    p_feedback.add_argument("--session-id-b64", default="")
    p_feedback.add_argument("--score", type=float, default=0.0)
    p_feedback.add_argument("--outcome", default="")
    p_feedback.add_argument("--outcome-b64", default="")
    p_feedback.add_argument(
        "--feedback-path",
        default="storage/artifacts/feedback/implicit_feedback.jsonl",
    )

    p_prefetch = sub.add_parser("prefetch-context")
    p_prefetch.add_argument("--npc-id", default="")
    p_prefetch.add_argument("--npc-id-b64", default="")
    p_prefetch.add_argument("--behavior-state", default="")
    p_prefetch.add_argument("--behavior-state-b64", default="")
    p_prefetch.add_argument("--location", default="")
    p_prefetch.add_argument("--location-b64", default="")
    p_prefetch.add_argument("--game-state-json", default="{}")
    p_prefetch.add_argument("--game-state-json-b64", default="")
    p_prefetch.add_argument("--top-predicted-states", type=int, default=2)
    p_prefetch.add_argument(
        "--transition-matrix-path",
        default="storage/artifacts/datasets/state_transition_matrix.json",
    )
    p_prefetch.add_argument(
        "--retrieval-corpus-path",
        default="data/retrieval_corpus_wide.jsonl",
    )
    p_prefetch.add_argument("--prefetch-top-docs-per-state", type=int, default=3)
    p_prefetch.add_argument("--max-prefetch-docs", type=int, default=8)
    p_prefetch.add_argument(
        "--prefetch-log-path",
        default="storage/artifacts/gspe/prefetch_events.jsonl",
    )

    p_load_trust = sub.add_parser("load-trust-score")
    p_load_trust.add_argument("--npc-id", default="")
    p_load_trust.add_argument("--npc-id-b64", default="")
    p_load_trust.add_argument("--player-id", default="")
    p_load_trust.add_argument("--player-id-b64", default="")
    p_load_trust.add_argument(
        "--trust-store-path",
        default="storage/artifacts/episodic_memory/trust_scores.json",
    )

    p_store_trust = sub.add_parser("store-trust-score")
    p_store_trust.add_argument("--npc-id", default="")
    p_store_trust.add_argument("--npc-id-b64", default="")
    p_store_trust.add_argument("--player-id", default="")
    p_store_trust.add_argument("--player-id-b64", default="")
    p_store_trust.add_argument("--session-id", default="")
    p_store_trust.add_argument("--session-id-b64", default="")
    p_store_trust.add_argument("--trust-score", type=float, default=0.0)
    p_store_trust.add_argument(
        "--trust-store-path",
        default="storage/artifacts/episodic_memory/trust_scores.json",
    )
    return parser


def run(argv: Optional[Sequence[str]] = None) -> Tuple[Dict[str, Any], int]:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "invalidate-prefix-cache":
        return handle_invalidate_prefix_cache(args), 0
    if args.command == "warm-prefix-cache":
        return handle_warm_prefix_cache(args), 0
    if args.command == "load-episodic":
        return handle_load_episodic(args), 0
    if args.command == "extract-episodic":
        return handle_extract_episodic(args), 0
    if args.command == "extract-episodic-interrupt":
        return handle_extract_episodic_interrupt(args), 0
    if args.command == "load-world-facts":
        return handle_load_world_facts(args), 0
    if args.command == "log-feedback":
        return handle_log_feedback(args), 0
    if args.command == "prefetch-context":
        return handle_prefetch_context(args), 0
    if args.command == "load-trust-score":
        return handle_load_trust_score(args), 0
    if args.command == "store-trust-score":
        return handle_store_trust_score(args), 0
    return {"ok": False, "error": f"unknown_command:{args.command}"}, 2


def main() -> None:
    payload, code = run()
    print(json.dumps(payload, ensure_ascii=False))
    raise SystemExit(code)


if __name__ == "__main__":
    main()

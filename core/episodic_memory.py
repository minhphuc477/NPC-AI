"""Persistent episodic memory store for NPC dialogue systems.

This module provides:
- JSONL-backed memory persistence across runs/sessions,
- lightweight lexical retrieval for relevant prior episodes,
- formatted memory snippets for prompt injection.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

TOKEN_RE = re.compile(r"[a-z0-9']+")


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall((text or "").lower())


def _clip(text: str, limit: int) -> str:
    raw = " ".join(str(text or "").strip().split())
    if len(raw) <= limit:
        return raw
    return raw[: max(0, limit - 3)].rstrip() + "..."


@dataclass
class EpisodicMemoryRecord:
    memory_id: str
    timestamp_utc: str
    npc_id: str
    persona: str
    behavior_state: str
    location: str
    player_input: str
    npc_response: str
    tags: List[str] = field(default_factory=list)
    source: str = ""
    run_id: str = ""

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "EpisodicMemoryRecord":
        return cls(
            memory_id=str(row.get("memory_id", "")).strip(),
            timestamp_utc=str(row.get("timestamp_utc", "")).strip() or utc_iso(),
            npc_id=str(row.get("npc_id", "")).strip(),
            persona=str(row.get("persona", "")).strip(),
            behavior_state=str(row.get("behavior_state", "")).strip(),
            location=str(row.get("location", "")).strip(),
            player_input=str(row.get("player_input", "")).strip(),
            npc_response=str(row.get("npc_response", "")).strip(),
            tags=[str(x).strip() for x in row.get("tags", []) if str(x).strip()],
            source=str(row.get("source", "")).strip(),
            run_id=str(row.get("run_id", "")).strip(),
        )

    def to_row(self) -> Dict[str, Any]:
        return asdict(self)

    def retrieval_text(self) -> str:
        parts = [
            self.persona,
            self.behavior_state,
            self.location,
            self.player_input,
            self.npc_response,
            " ".join(self.tags),
        ]
        return " ".join(p for p in parts if p).strip()

    def summary(self, max_player_chars: int = 90, max_response_chars: int = 120) -> str:
        head = []
        if self.behavior_state:
            head.append(self.behavior_state)
        if self.location:
            head.append(f"at {self.location}")
        prefix = " ".join(head).strip()
        player = _clip(self.player_input, max_player_chars)
        npc = _clip(self.npc_response, max_response_chars)
        if prefix:
            return f"{prefix}: Player '{player}' -> NPC '{npc}'"
        return f"Player '{player}' -> NPC '{npc}'"


class EpisodicMemoryStore:
    def __init__(self, path: Path, max_records: int = 4000) -> None:
        self.path = Path(path)
        self.max_records = max(1, int(max_records))
        self.records: List[EpisodicMemoryRecord] = []
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        self.records = []
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if not isinstance(row, dict):
                    continue
                rec = EpisodicMemoryRecord.from_row(row)
                if not rec.memory_id:
                    continue
                if not rec.npc_response:
                    continue
                self.records.append(rec)
        if len(self.records) > self.max_records:
            self.records = self.records[-self.max_records :]
            self._rewrite_file()

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def _next_id(self) -> str:
        self._ensure_loaded()
        return f"mem_{len(self.records) + 1:07d}"

    def _rewrite_file(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            for rec in self.records:
                handle.write(json.dumps(rec.to_row(), ensure_ascii=False) + "\n")

    def add_record(
        self,
        *,
        npc_id: str,
        persona: str,
        behavior_state: str,
        location: str,
        player_input: str,
        npc_response: str,
        tags: Optional[Sequence[str]] = None,
        source: str = "",
        run_id: str = "",
        timestamp_utc: str = "",
    ) -> EpisodicMemoryRecord:
        self._ensure_loaded()
        rec = EpisodicMemoryRecord(
            memory_id=self._next_id(),
            timestamp_utc=str(timestamp_utc).strip() or utc_iso(),
            npc_id=str(npc_id).strip(),
            persona=str(persona).strip(),
            behavior_state=str(behavior_state).strip(),
            location=str(location).strip(),
            player_input=str(player_input).strip(),
            npc_response=str(npc_response).strip(),
            tags=[str(x).strip() for x in (tags or []) if str(x).strip()],
            source=str(source).strip(),
            run_id=str(run_id).strip(),
        )
        if not rec.npc_response:
            return rec
        self.records.append(rec)
        if len(self.records) > self.max_records:
            self.records = self.records[-self.max_records :]
            self._rewrite_file()
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(rec.to_row(), ensure_ascii=False) + "\n")
        return rec

    @staticmethod
    def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
        sa = set(a)
        sb = set(b)
        if not sa or not sb:
            return 0.0
        return float(len(sa & sb)) / float(len(sa | sb))

    def retrieve(
        self,
        *,
        query: str,
        top_k: int = 3,
        min_score: float = 0.12,
        npc_id: str = "",
        behavior_state: str = "",
    ) -> List[Dict[str, Any]]:
        self._ensure_loaded()
        if not self.records:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        npc_id_norm = str(npc_id).strip().lower()
        state_norm = str(behavior_state).strip().lower()
        scored: List[tuple[float, EpisodicMemoryRecord]] = []
        for rec in self.records:
            rec_tokens = tokenize(rec.retrieval_text())
            lexical = self._jaccard(query_tokens, rec_tokens)
            if lexical <= 0.0:
                continue
            score = lexical
            if npc_id_norm and str(rec.npc_id).strip().lower() == npc_id_norm:
                score += 0.08
            if state_norm and str(rec.behavior_state).strip().lower() == state_norm:
                score += 0.05
            score = max(0.0, min(1.0, score))
            if score < float(min_score):
                continue
            scored.append((score, rec))

        scored.sort(key=lambda x: x[0], reverse=True)
        limited = scored[: max(1, int(top_k))]
        out: List[Dict[str, Any]] = []
        for score, rec in limited:
            out.append(
                {
                    "memory_id": rec.memory_id,
                    "score": float(score),
                    "timestamp_utc": rec.timestamp_utc,
                    "npc_id": rec.npc_id,
                    "persona": rec.persona,
                    "behavior_state": rec.behavior_state,
                    "location": rec.location,
                    "player_input": rec.player_input,
                    "npc_response": rec.npc_response,
                    "summary": rec.summary(),
                    "tags": list(rec.tags),
                    "source": rec.source,
                    "run_id": rec.run_id,
                }
            )
        return out

    def stats(self) -> Dict[str, Any]:
        self._ensure_loaded()
        by_npc: Dict[str, int] = {}
        by_state: Dict[str, int] = {}
        for rec in self.records:
            npc_key = rec.npc_id or "unknown"
            st_key = rec.behavior_state or "unknown"
            by_npc[npc_key] = int(by_npc.get(npc_key, 0)) + 1
            by_state[st_key] = int(by_state.get(st_key, 0)) + 1
        return {
            "path": str(self.path),
            "count": len(self.records),
            "max_records": self.max_records,
            "by_npc": dict(sorted(by_npc.items(), key=lambda kv: (-kv[1], kv[0]))),
            "by_behavior_state": dict(sorted(by_state.items(), key=lambda kv: (-kv[1], kv[0]))),
        }


def format_episodic_memories(memories: Sequence[Dict[str, Any]], max_items: int = 3) -> str:
    if not memories:
        return ""
    lines: List[str] = []
    for row in list(memories)[: max(1, int(max_items))]:
        summary = str(row.get("summary", "")).strip()
        if not summary:
            continue
        score = float(row.get("score", float("nan")))
        if math.isnan(score):
            lines.append(f"- {summary}")
        else:
            lines.append(f"- {summary} (score={score:.3f})")
    return "\n".join(lines).strip()

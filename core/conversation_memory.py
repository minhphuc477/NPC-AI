"""
BD-NSCA Conversation Memory System

Implements a Sliding Window + Summary memory manager for NPC conversations.
Each NPC maintains its own memory, enabling persistent multi-turn dialogue.

Architecture:
    - Recent turns kept in full text (configurable window size)
    - Older turns compressed into a summary sentence
    - Emotional state tracked across conversation
    - Trust drift computed from interaction history
    - Multilingual support (VI/EN)
"""
from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """A single player↔NPC exchange."""
    player: str
    npc: str
    timestamp: str = ""
    sentiment: float = 0.0  # -1.0 (hostile) to 1.0 (friendly)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class NPCMemory:
    """Per-NPC conversation memory state."""
    npc_id: str
    summary: str = ""
    recent_turns: List[ConversationTurn] = field(default_factory=list)
    trust_level: float = 50.0       # 0-100 scale
    total_interactions: int = 0
    emotional_valence: float = 0.0  # running average sentiment
    first_met: str = ""
    last_interaction: str = ""
    key_facts: List[str] = field(default_factory=list)  # player-revealed facts

    def __post_init__(self):
        if not self.first_met:
            self.first_met = datetime.now(timezone.utc).isoformat()


class ConversationMemory:
    """Manages conversation memory for all NPCs.

    Usage:
        memory = ConversationMemory(window_size=5)
        memory.add_turn("guard_01", "Hello!", "Halt! Who goes there?", 0.0)
        context = memory.get_memory_context("guard_01", language="vi")
    """

    def __init__(self, window_size: int = 5, max_summary_length: int = 200):
        self.window_size = window_size
        self.max_summary_length = max_summary_length
        self._memories: Dict[str, NPCMemory] = {}

    def _get_or_create(self, npc_id: str) -> NPCMemory:
        if npc_id not in self._memories:
            self._memories[npc_id] = NPCMemory(npc_id=npc_id)
        return self._memories[npc_id]

    def add_turn(
        self,
        npc_id: str,
        player_input: str,
        npc_response: str,
        sentiment: float = 0.0,
        language: str = "vi"
    ) -> None:
        """Record a conversation turn and update memory state."""
        mem = self._get_or_create(npc_id)
        turn = ConversationTurn(
            player=player_input,
            npc=npc_response,
            sentiment=sentiment,
        )
        mem.recent_turns.append(turn)
        mem.total_interactions += 1
        mem.last_interaction = turn.timestamp

        # Update trust based on sentiment
        trust_delta = sentiment * 3.0
        mem.trust_level = max(0.0, min(100.0, mem.trust_level + trust_delta))

        # Update running emotional valence (EMA)
        alpha = 0.3
        mem.emotional_valence = (
            alpha * sentiment + (1 - alpha) * mem.emotional_valence
        )

        # Extract key facts
        self._extract_facts(mem, player_input)

        # Compress if window exceeded
        if len(mem.recent_turns) > self.window_size:
            self._compress(mem, language)

    def _extract_facts(self, mem: NPCMemory, player_input: str) -> None:
        """Extract notable facts from player statements (Bi-lingual)."""
        fact_patterns = [
            # English
            ("I am ", "Player claims to be"),
            ("I'm ", "Player claims to be"),
            ("My name is ", "Player's name is"),
            ("I come from ", "Player comes from"),
            ("I need ", "Player needs"),
            ("I have ", "Player has"),
            # Vietnamese
            ("Tôi là ", "Người chơi tự nhận là"),
            ("Tôi tên ", "Tên người chơi là"),
            ("Tôi đến từ ", "Người chơi đến từ"),
            ("Tôi cần ", "Người chơi cần"),
            ("Tôi có ", "Người chơi có"),
            ("Mình là ", "Người chơi tự nhận là"),
        ]
        lower = player_input.lower()
        for pattern, prefix in fact_patterns:
            pattern_lower = pattern.lower()
            if pattern_lower in lower:
                # Find the actual case-sensitive start index
                try:
                    # Simple heuristic: scan case-insensitive
                    idx = lower.find(pattern_lower)
                    if idx >= 0:
                        content = player_input[idx + len(pattern):].strip().rstrip(".!")
                        if content and len(content) < 100:
                            fact = f"{prefix} {content}"
                            # Avoid duplicates
                            if fact not in mem.key_facts:
                                mem.key_facts.append(fact)
                                # Keep only last 10 facts
                                if len(mem.key_facts) > 10:
                                    mem.key_facts = mem.key_facts[-10:]
                except Exception:
                    continue
                break

    def _compress(self, mem: NPCMemory, language: str) -> None:
        """Compress oldest turns into summary."""
        overflow = mem.recent_turns[: -self.window_size]
        mem.recent_turns = mem.recent_turns[-self.window_size:]

        # Build summary from overflow turns
        summaries = []
        is_vi = (language == "vi")
        p_label = "Người chơi" if is_vi else "Player"
        n_label = "NPC"

        for turn in overflow:
            summaries.append(
                f"{p_label}: '{turn.player[:50]}' → {n_label}: '{turn.npc[:50]}'"
            )

        new_summary = "; ".join(summaries)
        
        # In a real implementation, we would call an LLM here to semantic summarize
        # For now, we append string summaries
        # self._consolidate_memory(mem, new_summary)
        
        if mem.summary:
            combined = f"{mem.summary}; {new_summary}"
        else:
            combined = new_summary

        # Truncate to max length
        if len(combined) > self.max_summary_length:
            combined = combined[-self.max_summary_length:]
            first_semi = combined.find(";")
            if first_semi > 0:
                combined = combined[first_semi + 1:].strip()

        mem.summary = combined

    def consolidate_memory(self, npc_id: str, llm_client=None):
        """
        [Advanced Codebase Improvement]
        Uses an LLM to semantically compress the 'summary' string into a concise narrative.
        This would be termed 'Episodic Memory Consolidation'.
        
        Args:
             npc_id: The NPC to consolidate
             llm_client: A function or client to call the LLM (e.g. Ollama)
        """
        mem = self._memories.get(npc_id)
        if not mem or not mem.summary:
            return

        if llm_client:
            prompt = f"Summarize this conversation history into one concise sentence relevant to the NPC:\n{mem.summary}"
            try:
                # Hypothetical synchronous call
                reduced_summary = llm_client(prompt)
                mem.summary = reduced_summary.strip()
            except Exception as e:
                logger.error(f"Memory consolidation failed: {e}")


    def get_memory_context(self, npc_id: str, language: str = "vi") -> str:
        """Generate formatted memory context for prompt injection.
        
        Args:
            npc_id: Target NPC
            language: 'vi' or 'en' for labels
        """
        mem = self._memories.get(npc_id)
        if not mem or mem.total_interactions == 0:
            return ""

        is_vi = (language == "vi")
        parts = []

        # Labels
        L_PREV = "Hội thoại trước:" if is_vi else "Previous conversations:"
        L_KNOWN = "Thông tin về người chơi:" if is_vi else "Known about player:"
        L_RECENT = "Hội thoại gần đây:" if is_vi else "Recent dialogue:"
        L_PLAYER = "Người chơi" if is_vi else "Player"
        L_RELATION = "Quan hệ: Tin tưởng={:.0f}/100 ({}), Tương tác={}" if is_vi \
                     else "Relationship: Trust={:.0f}/100 ({}), Interactions={}"

        # Conversation summary
        if mem.summary:
            parts.append(f"{L_PREV} {mem.summary}")

        # Key facts about the player
        if mem.key_facts:
            facts = ", ".join(mem.key_facts[-5:])
            parts.append(f"{L_KNOWN} {facts}")

        # Recent dialogue
        if mem.recent_turns:
            recent_lines = []
            # parts.append(L_RECENT) # Logic in prompt builder handles header if needed, 
                                     # but usually prompt builder adds 'Conversation History:' header.
                                     # We just return the content here.
            for turn in mem.recent_turns[-3:]:
                recent_lines.append(f"  {L_PLAYER}: {turn.player}")
                recent_lines.append(f"  NPC: {turn.npc}")
            parts.append(f"{L_RECENT}\n" + "\n".join(recent_lines))

        # Relationship state
        trust_label = self._trust_label(mem.trust_level, is_vi)
        parts.append(
            L_RELATION.format(mem.trust_level, trust_label, mem.total_interactions)
        )

        return "\n".join(parts)

    def get_trust_level(self, npc_id: str) -> float:
        mem = self._memories.get(npc_id)
        return mem.trust_level if mem else 50.0

    def get_emotional_valence(self, npc_id: str) -> float:
        mem = self._memories.get(npc_id)
        return mem.emotional_valence if mem else 0.0

    @staticmethod
    def _trust_label(trust: float, is_vi: bool) -> str:
        if trust >= 80:
            return "Đồng minh" if is_vi else "Trusted Ally"
        elif trust >= 60:
            return "Thân thiện" if is_vi else "Friendly"
        elif trust >= 40:
            return "Trung lập" if is_vi else "Neutral"
        elif trust >= 20:
            return "Nghi ngờ" if is_vi else "Suspicious"
        else:
            return "Thù địch" if is_vi else "Hostile"

    def save_state(self, filepath: str) -> None:
        state = {}
        for npc_id, mem in self._memories.items():
            mem_dict = asdict(mem)
            state[npc_id] = mem_dict
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def load_state(self, filepath: str) -> None:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                state = json.load(f)
            for npc_id, mem_dict in state.items():
                turns = [ConversationTurn(**t) for t in mem_dict.pop("recent_turns", [])]
                mem = NPCMemory(**mem_dict, recent_turns=turns)
                self._memories[npc_id] = mem
            logger.info(f"Loaded memory for {len(self._memories)} NPCs")
        except FileNotFoundError:
            logger.info(f"No saved memory at {filepath}, starting fresh")
        except Exception as e:
            logger.warning(f"Failed to load memory: {e}")

    def reset(self, npc_id: str = None) -> None:
        if npc_id:
            self._memories.pop(npc_id, None)
        else:
            self._memories.clear()

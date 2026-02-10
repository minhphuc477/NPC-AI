"""
BD-NSCA Prompt Builder v3

Constructs structured prompts for NPC dialogue generation with support for:
- Conversation memory injection (sliding window + summary)
- Emotional state integration
- Dynamic game state context
- Legacy format backward compatibility
- Full English/Vietnamese localization
"""
from datetime import datetime
from typing import Dict, List, Optional

# Chat template tokens (Phi-3 format)
_SYS = "<" + "|system|" + ">"
_END = "<" + "|end|" + ">"
_USR = "<" + "|user|" + ">"
_AST = "<" + "|assistant|" + ">"


class PromptBuilder:
    def __init__(self, use_advanced_format: bool = True):
        self.use_advanced_format = use_advanced_format

    def build_prompt(
        self,
        npc_data: Dict,
        game_state: Dict,
        player_input: str,
        memory_context: str = "",
        emotional_state: Optional[Dict[str, float]] = None,
        language: str = "vi",
    ) -> str:
        """Build a prompt with optional memory and emotional state injection.

        Args:
            npc_data: NPC persona, name, traits
            game_state: Current game world state
            player_input: What the player said
            memory_context: Formatted string from ConversationMemory
            emotional_state: Emotion axes dict {joy, anger, fear, trust, surprise}
            language: 'vi' or 'en' for prompt labels
        """
        if self.use_advanced_format:
            return self._build_advanced_prompt(
                npc_data, game_state, player_input, memory_context, emotional_state, language
            )
        else:
            return self._build_legacy_prompt(npc_data, game_state, player_input)

    def _build_advanced_prompt(
        self,
        npc_data: Dict,
        game_state: Dict,
        player_input: str,
        memory_context: str = "",
        emotional_state: Optional[Dict[str, float]] = None,
        language: str = "vi",
    ) -> str:
        """Construct the full structured prompt with all context layers."""
        is_vi = (language == "vi")
        
        # Localized Labels
        L_STATE = "**Trạng thái hiện tại:**" if is_vi else "**Current State:**"
        L_BEHAVIOR = "- Hành vi: {}" if is_vi else "- Behavior: {}"
        L_MOOD = "- Tâm trạng: {}" if is_vi else "- Mood: {}"
        L_HEALTH = "- Sức khỏe: {}" if is_vi else "- Health: {}"
        L_EMOTION = "- Cảm xúc: {} (cường độ {}%)" if is_vi else "- Emotional State: {} ({}% intensity)"
        
        L_ENV = "**Môi trường:**" if is_vi else "**Environment:**"
        L_LOC = "- Địa điểm: {}" if is_vi else "- Location: {}"
        L_TIME = "- Thời gian: {}" if is_vi else "- Time: {}"
        L_NEARBY = "- Xung quanh: {}" if is_vi else "- Nearby: {}"
        
        L_RELATION = "**Mối quan hệ:**" if is_vi else "**Relationship with Player:**"
        L_TRUST = "- Tin tưởng: {}/100" if is_vi else "- Trust: {}/100"
        
        L_HISTORY = "**Lịch sử trò chuyện:**" if is_vi else "**Conversation History:**"
        L_SCENARIO = "**Bối cảnh:**" if is_vi else "**Scenario:**"
        
        persona_key = "persona_vi" if is_vi else "persona_en"
        # Fallback to 'persona' if language specific key missing, then default text
        persona = npc_data.get(persona_key, npc_data.get("persona", "Bạn là một NPC hữu ích." if is_vi else "You are a helpful NPC."))
        
        npc_name = npc_data.get("name", "NPC")

        # Game state extraction (values should come localized from game_state if possible, 
        # or we rely on them being simple enough)
        behavior = game_state.get("behavior_state", "Idle")
        mood = game_state.get("mood_state", "Neutral")
        health = game_state.get("health_state", "Healthy")
        location = game_state.get("location", "Unknown")
        time_of_day = game_state.get("time_of_day", "Daytime")
        nearby = game_state.get("nearby_entities", "None")
        trust = game_state.get("trust_level", 50)
        scenario = game_state.get("scenario_plot", "No specific scenario.")

        # Build system message parts
        parts = []
        if is_vi:
            parts.append("Bạn là {}. {}".format(npc_name, persona))
        else:
            parts.append("You are {}. {}".format(npc_name, persona))
            
        parts.append("")
        parts.append(L_STATE)
        parts.append(L_BEHAVIOR.format(behavior))
        parts.append(L_MOOD.format(mood))
        parts.append(L_HEALTH.format(health))

        # Emotional state (if available)
        if emotional_state:
            dominant = max(emotional_state, key=emotional_state.get)
            intensity = emotional_state[dominant]
            parts.append(L_EMOTION.format(dominant, int(intensity * 100)))

        parts.append("")
        parts.append(L_ENV)
        parts.append(L_LOC.format(location))
        parts.append(L_TIME.format(time_of_day))
        parts.append(L_NEARBY.format(nearby))
        parts.append("")
        parts.append(L_RELATION)
        parts.append(L_TRUST.format(trust))

        # Conversation memory
        if memory_context:
            parts.append("")
            parts.append(L_HISTORY)
            parts.append(memory_context)

        parts.append("")
        parts.append(L_SCENARIO)
        parts.append(str(scenario))

        system_block = "\n".join(parts)

        # Assemble final prompt using Phi-3 chat template
        lines = [
            _SYS,
            system_block,
            _END,
            _USR,
            player_input,
            _AST,
        ]
        return "\n".join(lines) + "\n"

    def _build_legacy_prompt(
        self, npc_data: Dict, game_state: Dict, player_input: str
    ) -> str:
        """Backward-compatible simple format."""
        persona = npc_data.get("persona", "You are a helpful NPC.")
        npc_id = npc_data.get("id", "NPC")
        scenario = game_state.get("scenario", "")
        return "System: {}\nName: {}\nContext: {}\n\nQuestion: {}\nAnswer:".format(
            persona, npc_id, scenario, player_input
        )

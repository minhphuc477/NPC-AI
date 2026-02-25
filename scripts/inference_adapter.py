#!/usr/bin/env python3
"""BD-NSCA Inference Adapter for Ollama.

Integrates PromptBuilder formatting with a response-control layer to reduce
template/meta artifacts and improve context-grounded dialogue quality.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import requests

# Add project root to import core modules.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.prompt_builder import PromptBuilder
from core.response_controller import ControlConfig, control_response, sanitize_response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

_TOKEN_RE = re.compile(r"[a-z0-9']+")
_STOPWORDS = {
    "you",
    "are",
    "the",
    "and",
    "with",
    "that",
    "this",
    "your",
    "from",
    "into",
    "about",
    "while",
    "who",
    "for",
    "can",
    "will",
    "game",
    "npc",
}
_PERSONA_TERMS = {
    "strict",
    "fair",
    "brief",
    "suspicious",
    "talkative",
    "profit",
    "calm",
    "caring",
    "precise",
    "direct",
    "practical",
    "formal",
    "sources",
    "precision",
    "mysterious",
    "indirect",
    "procedural",
    "scholar",
    "healer",
    "guard",
    "merchant",
}


@dataclass
class InferenceConfig:
    model: str = "phi3:mini"
    temperature: float = 0.7
    max_tokens: int = 256
    timeout: float = 30.0
    use_json_prompt: bool = False
    use_response_control: bool = True
    control_min_context_coverage: float = 0.25
    control_min_persona_coverage: float = 0.15
    control_rewrite_temperature: float = 0.2
    control_rewrite_max_tokens: int = 96
    control_rewrite_candidates: int = 3
    control_rewrite_temperature_step: float = 0.15
    control_allow_best_effort_rewrite: bool = True
    disable_control_rewrite: bool = False


class BDNSCAInference:
    """Inference client for BD-NSCA architecture using Ollama."""

    def __init__(self, config: Optional[InferenceConfig] = None, base_url: str = OLLAMA_URL):
        self.config = config or InferenceConfig()
        self.base_url = base_url.rstrip("/")
        self.prompt_builder = PromptBuilder(
            use_advanced_format=True,
            use_json_format=bool(self.config.use_json_prompt),
        )

    @staticmethod
    def _dedupe_keywords(candidates: Sequence[str], max_items: int) -> List[str]:
        out: List[str] = []
        seen = set()
        for raw in candidates:
            candidate = " ".join(str(raw).strip().lower().split())
            if not candidate:
                continue
            if candidate in seen:
                continue
            seen.add(candidate)
            out.append(candidate)
            if len(out) >= max_items:
                break
        return out

    @staticmethod
    def _extract_json_dialogue_text(text: str) -> str:
        raw = (text or "").strip()
        if not raw:
            return ""

        if raw.startswith("```"):
            raw = raw.strip("`").strip()
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()

        if raw.startswith("{") and raw.endswith("}"):
            try:
                payload = json.loads(raw)
                if isinstance(payload, dict):
                    value = payload.get("text", "")
                    if isinstance(value, str):
                        return value.strip()
            except Exception:
                return text
        return text

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return _TOKEN_RE.findall((text or "").lower())

    def _extract_persona_keywords(self, persona: str) -> List[str]:
        p = (persona or "").lower()
        candidates: List[str] = []
        for term in _PERSONA_TERMS:
            if term in p:
                candidates.append(term)

        chunks = re.split(r"[,:;]", p)
        for chunk in chunks:
            tokens = [t for t in self._tokenize(chunk) if t not in _STOPWORDS]
            if 1 <= len(tokens) <= 4:
                candidates.append(" ".join(tokens))
        return self._dedupe_keywords(candidates, max_items=10)

    def _extract_context_keywords(self, context: Dict[str, str]) -> List[str]:
        candidates: List[str] = []
        for key, value in context.items():
            key_text = str(key).replace("_", " ").strip().lower()
            if key_text and key_text not in _STOPWORDS:
                candidates.append(key_text)

            value_text = str(value).strip().lower()
            if not value_text:
                continue
            fragments = re.split(r"[;,.]", value_text)
            for fragment in fragments:
                tokens = [t for t in self._tokenize(fragment) if t not in _STOPWORDS]
                if 1 <= len(tokens) <= 6:
                    candidates.append(" ".join(tokens))
        return self._dedupe_keywords(candidates, max_items=12)

    @staticmethod
    def _context_to_string(context: Dict[str, str]) -> str:
        parts = []
        for key, value in context.items():
            key_s = str(key).strip()
            value_s = str(value).strip()
            if key_s and value_s:
                parts.append(f"{key_s}={value_s}")
        return "; ".join(parts)

    def _request_generation(self, prompt: str, temperature: float, max_tokens: int) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "stop": ["<|user|>", "<|model|>", "<|end|>"],
            },
        }
        try:
            resp = requests.post(url, json=payload, timeout=self.config.timeout)
            resp.raise_for_status()
            data = resp.json()
            return str(data.get("response", ""))
        except requests.RequestException as exc:
            logger.error("Ollama request failed: %s", exc)
            return ""
        except ValueError as exc:
            logger.error("Failed to parse Ollama response: %s", exc)
            return ""

    def format_prompt(
        self,
        persona: str,
        plot: str,
        context: Dict[str, str],
        player_input: str,
        npc_name: str = "NPC",
        language: str = "vi",
    ) -> str:
        """Format a BD-NSCA prompt using the unified PromptBuilder."""
        npc_data = {
            "name": npc_name,
            "persona": persona,
        }

        game_state = context.copy()
        game_state["scenario_plot"] = plot

        # Compatibility mapping for older field names.
        if "Trạng thái" in context and "behavior_state" not in context:
            game_state["behavior_state"] = context["Trạng thái"]
        if "Tâm trạng" in context and "mood_state" not in context:
            game_state["mood_state"] = context["Tâm trạng"]

        return self.prompt_builder.build_prompt(
            npc_data=npc_data,
            game_state=game_state,
            player_input=player_input,
            memory_context="",
            emotional_state=None,
            language=language,
        )

    def generate(
        self,
        persona: str,
        plot: str,
        context: Dict[str, str],
        player_input: str,
        npc_name: str = "NPC",
        language: str = "vi",
    ) -> str:
        """Generate an NPC response using Ollama + optional response control."""
        prompt = self.format_prompt(persona, plot, context, player_input, npc_name, language)
        raw_response = self._request_generation(
            prompt=prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        if not raw_response:
            return ""

        extracted = self._extract_json_dialogue_text(raw_response)
        if not self.config.use_response_control:
            cleaned = sanitize_response(extracted)
            return cleaned or sanitize_response(raw_response)

        context_keywords = self._extract_context_keywords(context)
        persona_keywords = self._extract_persona_keywords(persona)
        dynamic_context = self._context_to_string(context)
        config = ControlConfig(
            min_context_coverage=self.config.control_min_context_coverage if context_keywords else 0.0,
            min_persona_coverage=self.config.control_min_persona_coverage if persona_keywords else 0.0,
            rewrite_temperature=self.config.control_rewrite_temperature,
            rewrite_max_tokens=self.config.control_rewrite_max_tokens,
            rewrite_candidates=self.config.control_rewrite_candidates,
            rewrite_temperature_step=self.config.control_rewrite_temperature_step,
            enable_rewrite=not self.config.disable_control_rewrite,
            allow_best_effort_rewrite=self.config.control_allow_best_effort_rewrite,
        )

        def rewrite_fn(rewrite_prompt: str, rewrite_max_tokens: int, rewrite_temperature: float) -> str:
            return self._request_generation(
                prompt=rewrite_prompt,
                temperature=rewrite_temperature,
                max_tokens=rewrite_max_tokens,
            )

        controlled = control_response(
            raw_response=extracted,
            persona=persona,
            dynamic_context=dynamic_context,
            player_input=player_input,
            context_keywords=context_keywords,
            persona_keywords=persona_keywords,
            rewrite_fn=rewrite_fn,
            config=config,
        )

        if controlled.repaired:
            logger.debug(
                "Response repaired (%s) source=%s",
                controlled.repair_reason,
                controlled.source,
            )
        return controlled.response

    def health_check(self) -> bool:
        """Check whether the Ollama server is reachable."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.ok
        except requests.RequestException:
            return False


def test_inference() -> bool:
    """Test the inference adapter with sample data."""
    inference = BDNSCAInference()
    if not inference.health_check():
        logger.warning("Ollama server is not reachable. Running prompt formatting test only.")

    persona = "You are the Gatekeeper: strict, fair, and cautious."
    plot = "The player asks to enter during a high-alert night watch."
    context = {
        "behavior_state": "Guarding",
        "health_state": "Healthy",
        "mood_state": "Suspicious",
        "location": "Village Gate",
        "time_of_day": "Night",
    }
    player_input = "Let me inside the village now."

    logger.info("Testing BD-NSCA inference adapter...")
    formatted = inference.format_prompt(persona, plot, context, player_input, npc_name="Gatekeeper")
    logger.info("Generated prompt:\n%s", formatted)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BD-NSCA Inference Adapter")
    parser.add_argument("--test", action="store_true", help="Run adapter test")
    parser.add_argument("--model", default="phi3:mini", help="Ollama model name")
    args = parser.parse_args()

    if args.test:
        test_inference()

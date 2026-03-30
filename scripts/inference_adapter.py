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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import requests

# Add project root to import core modules.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.prompt_builder import PromptBuilder
from core.episodic_memory import EpisodicMemoryStore, format_episodic_memories
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
    control_min_context_coverage: float = 0.33
    control_min_persona_coverage: float = 0.18
    control_rewrite_temperature: float = 0.2
    control_rewrite_max_tokens: int = 96
    control_rewrite_candidates: int = 2
    control_rewrite_temperature_step: float = 0.15
    control_rewrite_budget_ms: float = 0.0
    control_rewrite_budget_multiplier: float = 0.0
    control_allow_best_effort_rewrite: bool = True
    control_behavior_adaptation_enabled: bool = True
    control_adaptive_acceptance_enabled: bool = True
    control_adaptive_candidate_score: float = 0.38
    control_adaptive_context_coverage: float = 0.14
    control_adaptive_persona_coverage: float = 0.10
    control_adaptive_high_confidence_score: float = 0.53
    control_adaptive_mid_confidence_score: float = 0.40
    control_adaptive_high_confidence_rewrites: int = 1
    control_adaptive_mid_confidence_rewrites: int = 2
    control_adaptive_low_confidence_rewrites: int = 3
    control_low_confidence_retry_requires_gain: bool = True
    control_low_confidence_retry_min_score_gain: float = 0.01
    control_low_confidence_retry_min_coverage_gain: float = 0.02
    control_intent_risk_adaptation_enabled: bool = True
    control_latency_adaptation_enabled: bool = False
    control_latency_relax_start_pressure: float = 0.55
    control_latency_relax_max_delta: float = 0.12
    control_intent_focused_context_enabled: bool = True
    control_intent_focus_min_keep: int = 3
    control_intent_focus_keep_ratio_low: float = 0.45
    control_intent_focus_keep_ratio_medium: float = 0.65
    control_intent_focus_keep_ratio_high: float = 1.0
    control_intent_focus_min_relevance: float = 0.20
    control_near_pass_enabled: bool = True
    control_near_pass_max_context_gap: float = 0.05
    control_near_pass_max_persona_gap: float = 0.04
    control_near_pass_score_floor: float = 0.34
    control_near_pass_block_high_risk: bool = True
    disable_control_rewrite: bool = False
    enable_draft_then_score: bool = False
    draft_max_tokens: int = 20
    draft_max_resamples: int = 1
    draft_score_threshold: float = 0.36
    draft_temperature_jitter: float = 0.08
    draft_only_with_response_control: bool = True
    enable_episodic_memory: bool = False
    episodic_memory_path: str = "storage/artifacts/episodic_memory/inference_adapter_memory.jsonl"
    episodic_memory_top_k: int = 3
    episodic_memory_min_score: float = 0.12
    episodic_memory_max_records: int = 4000


class BDNSCAInference:
    """Inference client for BD-NSCA architecture using Ollama."""

    def __init__(self, config: Optional[InferenceConfig] = None, base_url: str = OLLAMA_URL):
        self.config = config or InferenceConfig()
        self.base_url = base_url.rstrip("/")
        self.prompt_builder = PromptBuilder(
            use_advanced_format=True,
            use_json_format=bool(self.config.use_json_prompt),
        )
        self.episodic_store: Optional[EpisodicMemoryStore] = None
        if bool(self.config.enable_episodic_memory):
            self.episodic_store = EpisodicMemoryStore(
                path=Path(str(self.config.episodic_memory_path)),
                max_records=max(100, int(self.config.episodic_memory_max_records)),
            )
            self.episodic_store.load()

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

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _keyword_coverage(self, text: str, keywords: Sequence[str]) -> float:
        kw = [str(x).strip().lower() for x in keywords if str(x).strip()]
        if not kw:
            return 0.0
        lowered = (text or "").lower()
        tokens = set(self._tokenize(text))
        hits = 0
        for item in kw:
            if item in lowered:
                hits += 1
                continue
            item_tokens = self._tokenize(item)
            if item_tokens and all(tok in tokens for tok in item_tokens):
                hits += 1
        return float(hits) / float(len(kw))

    def _draft_behavior_bucket(self, behavior_state: str) -> str:
        state = " ".join(str(behavior_state or "").strip().lower().replace("_", " ").split())
        if not state:
            return "general"
        if any(token in state for token in ("detain", "negotiat", "investigat", "combat", "guard", "deception")):
            return "conflict"
        if any(token in state for token in ("assist", "quest", "handoff", "craft", "trade", "repair", "treat", "research", "forg")):
            return "task"
        if any(token in state for token in ("patrol", "social", "idle", "chat", "observe")):
            return "social"
        return "general"

    def _draft_threshold_for_state(self, behavior_state: str) -> float:
        threshold = float(self.config.draft_score_threshold)
        bucket = self._draft_behavior_bucket(behavior_state)
        if bucket == "conflict":
            threshold += 0.08
        elif bucket == "task":
            threshold -= 0.02
        elif bucket == "social":
            threshold -= 0.05
        return self._clamp01(max(0.15, min(0.95, threshold)))

    def _draft_naturalness(self, text: str) -> float:
        words = self._tokenize(text)
        wc = len(words)
        if wc <= 0:
            return 0.0
        distinct = len(set(words)) / float(wc)
        target = 26.0
        spread = 18.0
        length_score = max(0.0, 1.0 - abs(float(wc) - target) / spread)
        return self._clamp01(0.65 * distinct + 0.35 * length_score)

    def _draft_quality_score(
        self,
        draft_text: str,
        *,
        persona: str,
        dynamic_context: str,
        context_keywords: Sequence[str],
        persona_keywords: Sequence[str],
    ) -> float:
        text = sanitize_response(draft_text or "")
        if not text:
            return 0.0
        context_cov = self._keyword_coverage(text, context_keywords) if context_keywords else 0.0
        persona_cov = self._keyword_coverage(text, persona_keywords) if persona_keywords else 0.0
        naturalness = self._draft_naturalness(text)
        token_count = len(self._tokenize(text))
        length_norm = self._clamp01(float(token_count) / 20.0)
        score = 0.45 * context_cov + 0.30 * persona_cov + 0.20 * naturalness + 0.05 * length_norm

        lowered = text.lower()
        penalties = 0.0
        if token_count < 5:
            penalties += 0.20
        if any(marker in lowered for marker in ("as an ai", "assistant", "system prompt", "constraints:", "draft response:")):
            penalties += 0.22
        if dynamic_context and context_keywords and context_cov < 0.05:
            penalties += 0.08
        return self._clamp01(score - penalties)

    def _draft_temperature_for_attempt(self, attempt_idx: int) -> float:
        if attempt_idx <= 0:
            return float(self.config.temperature)
        step = (attempt_idx + 1) // 2
        sign = 1.0 if (attempt_idx % 2 == 1) else -1.0
        return float(self.config.temperature) + sign * float(self.config.draft_temperature_jitter) * float(step)

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
        memory_context: str = "",
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
            memory_context=memory_context,
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
        context_keywords = self._extract_context_keywords(context)
        persona_keywords = self._extract_persona_keywords(persona)
        dynamic_context = self._context_to_string(context)
        behavior_state = (
            str(
                context.get("behavior_state")
                or context.get("BehaviorTreeState")
                or context.get("behaviortreestate")
                or ""
            ).strip()
        )
        location = str(context.get("location", "")).strip()

        memory_context = ""
        if self.episodic_store is not None:
            query = " ".join([persona, player_input, dynamic_context, behavior_state, location]).strip()
            retrieved = self.episodic_store.retrieve(
                query=query,
                top_k=max(1, int(self.config.episodic_memory_top_k)),
                min_score=max(0.0, min(1.0, float(self.config.episodic_memory_min_score))),
                npc_id=npc_name,
                behavior_state=behavior_state,
            )
            memory_context = format_episodic_memories(retrieved, max_items=max(1, int(self.config.episodic_memory_top_k)))
            if memory_context:
                logger.debug("Episodic memory hits=%d for npc=%s", len(retrieved), npc_name)

        prompt = self.format_prompt(
            persona,
            plot,
            context,
            player_input,
            npc_name,
            language,
            memory_context=memory_context,
        )

        generation_temperature = float(self.config.temperature)
        draft_enabled = bool(self.config.enable_draft_then_score) and (
            bool(self.config.use_response_control) or not bool(self.config.draft_only_with_response_control)
        )
        if draft_enabled:
            attempts = max(1, int(self.config.draft_max_resamples) + 1)
            threshold = self._draft_threshold_for_state(behavior_state)
            accepted = False
            for attempt_idx in range(attempts):
                attempt_temp = max(0.0, min(1.5, self._draft_temperature_for_attempt(attempt_idx)))
                generation_temperature = attempt_temp
                draft = self._request_generation(
                    prompt=prompt,
                    temperature=attempt_temp,
                    max_tokens=max(1, int(self.config.draft_max_tokens)),
                )
                score = self._draft_quality_score(
                    draft,
                    persona=persona,
                    dynamic_context=dynamic_context,
                    context_keywords=context_keywords,
                    persona_keywords=persona_keywords,
                )
                if score >= threshold:
                    accepted = True
                    break
            logger.debug(
                "Draft prefilter: enabled=%s accepted=%s attempts=%d threshold=%.3f selected_temp=%.3f state=%s",
                draft_enabled,
                accepted,
                attempts,
                threshold,
                generation_temperature,
                behavior_state or "unknown",
            )

        raw_start_ns = time.perf_counter_ns()
        raw_response = self._request_generation(
            prompt=prompt,
            temperature=generation_temperature,
            max_tokens=self.config.max_tokens,
        )
        raw_elapsed_ms = max(0.0, (time.perf_counter_ns() - raw_start_ns) / 1_000_000.0)
        if not raw_response:
            return ""

        extracted = self._extract_json_dialogue_text(raw_response)
        if not self.config.use_response_control:
            cleaned = sanitize_response(extracted)
            final_response = cleaned or sanitize_response(raw_response)
            if self.episodic_store is not None and final_response:
                self.episodic_store.add_record(
                    npc_id=npc_name,
                    persona=persona,
                    behavior_state=behavior_state,
                    location=location,
                    player_input=player_input,
                    npc_response=final_response,
                    tags=[plot, str(context.get("time_of_day", "")).strip()],
                    source="inference_adapter",
                )
            return final_response

        config = ControlConfig(
            min_context_coverage=self.config.control_min_context_coverage if context_keywords else 0.0,
            min_persona_coverage=self.config.control_min_persona_coverage if persona_keywords else 0.0,
            rewrite_temperature=self.config.control_rewrite_temperature,
            rewrite_max_tokens=self.config.control_rewrite_max_tokens,
            rewrite_candidates=self.config.control_rewrite_candidates,
            rewrite_temperature_step=self.config.control_rewrite_temperature_step,
            enable_rewrite=not self.config.disable_control_rewrite,
            allow_best_effort_rewrite=self.config.control_allow_best_effort_rewrite,
            behavior_adaptation_enabled=self.config.control_behavior_adaptation_enabled,
            adaptive_acceptance_enabled=self.config.control_adaptive_acceptance_enabled,
            adaptive_candidate_score=self.config.control_adaptive_candidate_score,
            adaptive_context_coverage=self.config.control_adaptive_context_coverage,
            adaptive_persona_coverage=self.config.control_adaptive_persona_coverage,
            adaptive_high_confidence_score=self.config.control_adaptive_high_confidence_score,
            adaptive_mid_confidence_score=self.config.control_adaptive_mid_confidence_score,
            adaptive_high_confidence_rewrites=self.config.control_adaptive_high_confidence_rewrites,
            adaptive_mid_confidence_rewrites=self.config.control_adaptive_mid_confidence_rewrites,
            adaptive_low_confidence_rewrites=self.config.control_adaptive_low_confidence_rewrites,
            low_confidence_retry_requires_gain=self.config.control_low_confidence_retry_requires_gain,
            low_confidence_retry_min_score_gain=self.config.control_low_confidence_retry_min_score_gain,
            low_confidence_retry_min_coverage_gain=self.config.control_low_confidence_retry_min_coverage_gain,
            intent_risk_adaptation_enabled=self.config.control_intent_risk_adaptation_enabled,
            latency_adaptation_enabled=self.config.control_latency_adaptation_enabled,
            latency_relax_start_pressure=self.config.control_latency_relax_start_pressure,
            latency_relax_max_delta=self.config.control_latency_relax_max_delta,
            intent_focused_context_enabled=self.config.control_intent_focused_context_enabled,
            intent_focus_min_keep=self.config.control_intent_focus_min_keep,
            intent_focus_keep_ratio_low=self.config.control_intent_focus_keep_ratio_low,
            intent_focus_keep_ratio_medium=self.config.control_intent_focus_keep_ratio_medium,
            intent_focus_keep_ratio_high=self.config.control_intent_focus_keep_ratio_high,
            intent_focus_min_relevance=self.config.control_intent_focus_min_relevance,
            near_pass_enabled=self.config.control_near_pass_enabled,
            near_pass_max_context_gap=self.config.control_near_pass_max_context_gap,
            near_pass_max_persona_gap=self.config.control_near_pass_max_persona_gap,
            near_pass_score_floor=self.config.control_near_pass_score_floor,
            near_pass_block_high_risk=self.config.control_near_pass_block_high_risk,
        )

        rewrite_state: Dict[str, Any] = {
            "spent_ms": 0.0,
            "budget_exhausted": False,
        }

        def rewrite_fn(rewrite_prompt: str, rewrite_max_tokens: int, rewrite_temperature: float) -> str:
            budget_ms = float("inf")
            if self.config.control_rewrite_budget_ms > 0.0:
                budget_ms = min(budget_ms, float(self.config.control_rewrite_budget_ms))
            if self.config.control_rewrite_budget_multiplier > 0.0 and raw_elapsed_ms > 0.0:
                budget_ms = min(budget_ms, float(self.config.control_rewrite_budget_multiplier) * raw_elapsed_ms)
            spent_ms = float(rewrite_state.get("spent_ms", 0.0))
            if spent_ms >= budget_ms:
                rewrite_state["budget_exhausted"] = True
                return ""
            start_ns = time.perf_counter_ns()
            out = self._request_generation(
                prompt=rewrite_prompt,
                temperature=rewrite_temperature,
                max_tokens=rewrite_max_tokens,
            )
            rewrite_state["spent_ms"] = spent_ms + max(
                0.0,
                (time.perf_counter_ns() - start_ns) / 1_000_000.0,
            )
            return out

        controlled = control_response(
            raw_response=extracted,
            persona=persona,
            dynamic_context=dynamic_context,
            player_input=player_input,
            context_keywords=context_keywords,
            persona_keywords=persona_keywords,
            rewrite_fn=rewrite_fn,
            config=config,
            behavior_state=behavior_state,
            raw_latency_ms=raw_elapsed_ms,
            timeout_s=float(self.config.timeout),
        )

        if controlled.repaired:
            logger.debug(
                "Response repaired (%s) source=%s",
                controlled.repair_reason,
                controlled.source,
            )
        final_response = controlled.response
        if self.episodic_store is not None and final_response:
            self.episodic_store.add_record(
                npc_id=npc_name,
                persona=persona,
                behavior_state=behavior_state,
                location=location,
                player_input=player_input,
                npc_response=final_response,
                tags=[plot, str(context.get("time_of_day", "")).strip()],
                source="inference_adapter",
            )
        return final_response

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

#!/usr/bin/env python3
"""BD-NSCA Inference Adapter for Ollama.

Integrates with core.PromptBuilder to ensure inference prompts match 
training data exactly.
"""
import argparse
import json
import logging
import os
import sys
import requests
from dataclasses import dataclass
from typing import Dict, Optional
from pathlib import Path

# Add parent dir to path to import core modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.prompt_builder import PromptBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")


@dataclass
class InferenceConfig:
    model: str = "phi3:mini"
    temperature: float = 0.7
    max_tokens: int = 256
    timeout: float = 30.0


class BDNSCAInference:
    """Inference client for BD-NSCA architecture using Ollama."""
    
    def __init__(self, config: Optional[InferenceConfig] = None, base_url: str = OLLAMA_URL):
        self.config = config or InferenceConfig()
        self.base_url = base_url.rstrip("/")
        self.prompt_builder = PromptBuilder(use_advanced_format=True, use_json_format=True)
    
    def format_prompt(
        self, 
        persona: str, 
        plot: str, 
        context: Dict[str, str], 
        player_input: str,
        npc_name: str = "NPC",
        language: str = "vi"
    ) -> str:
        """Format BD-NSCA prompt using the unified PromptBuilder."""
        
        # Map flat arguments to PromptBuilder structure
        npc_data = {
            "name": npc_name,
            "persona": persona
        }
        
        # Inject plot into game state for the builder
        game_state = context.copy()
        game_state["scenario_plot"] = plot
        
        # Map context keys to standard keys if they are coming from legacy C++ calls
        # If the C++ side sends "Trạng thái" instead of "behavior_state", we might need mapping.
        # However, we are moving towards standard keys. 
        # For now, we assume the C++ or caller provides compatible keys or we rely on PB's flexibility.
        # But PromptBuilder V3 expects specific keys like 'behavior_state' to look up values.
        # If 'context' contains arbitary keys, PB won't show them unless we modify PB.
        # WE MUST ENSURE CONTEXT HAS STANDARD KEYS or map them.
        
        # Simple mapping for legacy calls (if any)
        if "Trạng thái" in context and "behavior_state" not in context:
            game_state["behavior_state"] = context["Trạng thái"]
        if "Tâm trạng" in context and "mood_state" not in context:
            game_state["mood_state"] = context["Tâm trạng"]
        
        return self.prompt_builder.build_prompt(
            npc_data=npc_data,
            game_state=game_state,
            player_input=player_input,
            memory_context="", # Memory usually injected by the caller or managed here? 
                               # For simple adapter, we assume memory is inside context or not used yet.
                               # Real implementation would call ConversationMemory here.
            emotional_state=None,
            language=language
        )
    
    def generate(
        self, 
        persona: str, 
        plot: str, 
        context: Dict[str, str], 
        player_input: str,
        npc_name: str = "NPC",
        language: str = "vi"
    ) -> str:
        """Generate NPC response using Ollama."""
        prompt = self.format_prompt(persona, plot, context, player_input, npc_name, language)
        
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
                "stop": ["<|user|>", "<|model|>", "<|end|>"]
            }
        }
        
        try:
            resp = requests.post(url, json=payload, timeout=self.config.timeout)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "")
        except requests.RequestException as e:
            logger.error(f"Ollama request failed: {e}")
            return ""
    
    def health_check(self) -> bool:
        """Check if Ollama server is running."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            # If 404 but connected, it might be the tags endpoint is different or server is up.
            # Usually /api/tags is good.
            return resp.ok
        except requests.RequestException:
            return False


def test_inference():
    """Test the inference adapter with sample data."""
    inference = BDNSCAInference()
    
    # Check if server is up, but don't fail, just warn
    if not inference.health_check():
        logger.warning("Ollama server not running or not reachable. Proceeding with prompt check only.")
    
    persona = "Bạn là Thần giữ cổng, một chiến binh già nghiêm khắc nhưng công bằng."
    plot = "Người chơi đến cổng làng trong lúc làng đang căng thẳng vì bọn cướp."
    # Using standard keys for V3
    context = {
        "behavior_state": "Đang canh gác (Guarding)",
        "health_state": "Khỏe mạnh (Healthy)",
        "mood_state": "Nghi ngờ (Suspicious)",
        "location": "Cổng làng",
        "time_of_day": "Buổi sáng"
    }
    player_input = "Xin chào, tôi muốn vào làng."
    
    logger.info("Testing BD-NSCA inference adapter...")
    formatted = inference.format_prompt(persona, plot, context, player_input, npc_name="Gatekeeper")
    logger.info("Generated Prompt View:\n%s", formatted)
    
    # We won't call generate() in test unless we know server is up to avoid hanging
    # response = inference.generate(persona, plot, context, player_input)
    # logger.info(f"NPC Response: {response}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BD-NSCA Inference Adapter")
    parser.add_argument("--test", action="store_true", help="Run test inference")
    parser.add_argument("--model", default="phi3:mini", help="Ollama model name")
    args = parser.parse_args()
    
    if args.test:
        test_inference()

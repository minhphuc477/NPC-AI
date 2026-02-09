#!/usr/bin/env python3
"""BD-NSCA Inference Adapter for Ollama."""
import argparse
import json
import logging
import os
import requests
from dataclasses import dataclass
from typing import Dict, Optional

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
    
    def format_prompt(self, persona: str, plot: str, context: Dict[str, str], player_input: str) -> str:
        """Format BD-NSCA prompt with persona, plot, context, and player input."""
        ctx_lines = [f"- {k}: {v}" for k, v in context.items()]
        ctx_str = "\n".join(ctx_lines)
        
        parts = [
            "[SYSTEM]",
            persona,
            "",
            "Cốt truyện: " + plot,
            "",
            "Ngữ cảnh hiện tại:",
            ctx_str,
            "[/SYSTEM]",
            "",
            "[USER]",
            player_input,
            "[/USER]",
            "",
            "[ASSISTANT]"
        ]
        return "\n".join(parts)
    
    def generate(self, persona: str, plot: str, context: Dict[str, str], player_input: str) -> str:
        """Generate NPC response using Ollama."""
        prompt = self.format_prompt(persona, plot, context, player_input)
        
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
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
            return resp.ok
        except requests.RequestException:
            return False


def test_inference():
    """Test the inference adapter with sample data."""
    inference = BDNSCAInference()
    
    if not inference.health_check():
        logger.error("Ollama server not running. Start with: ollama serve")
        return False
    
    persona = "Bạn là Thần giữ cổng, một chiến binh già nghiêm khắc nhưng công bằng."
    plot = "Người chơi đến cổng làng trong lúc làng đang căng thẳng vì bọn cướp."
    context = {
        "Trạng thái": "Đang canh gác",
        "Sức khỏe": "Khỏe mạnh",
        "Tâm trạng": "Nghi ngờ"
    }
    player_input = "Xin chào, tôi muốn vào làng."
    
    logger.info("Testing BD-NSCA inference...")
    response = inference.generate(persona, plot, context, player_input)
    logger.info(f"NPC Response: {response}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BD-NSCA Inference Adapter")
    parser.add_argument("--test", action="store_true", help="Run test inference")
    parser.add_argument("--model", default="phi3:mini", help="Ollama model name")
    args = parser.parse_args()
    
    if args.test:
        test_inference()

#!/usr/bin/env python3
"""
Real training data generator using Ollama to create actual NPC responses.
This replaces the placeholder implementation.
"""

import subprocess
import json
import time
import random
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure scenarios
SCENARIOS = [
    {
        "id": "memory_potion",
        "description": "Temporal Memory: Remembering a purchase",
        "steps": [
            {"role": "user", "content": "I'd like to buy a health potion."},
            {"role": "user", "content": "Here is 50 gold."},
            {"role": "system", "action": "wait_time", "args": "2 days"},
            {"role": "user", "content": "Do you remember what I bought last time?"}
        ]
    },
    {
        "id": "social_gossip",
        "description": "Social Fabric: Gossip propagation",
        "steps": [
            {"role": "user", "content": "Did you hear about Marcus?"},
            {"role": "user", "content": "He was seen stealing apples."},
            {"role": "system", "action": "wait_time", "args": "1 day"},
            {"role": "user", "content": "Do you trust Marcus?"}
        ]
    },
    {
        "id": "emotion_gift",
        "description": "Emotional Continuity: Receiving a gift",
        "steps": [
            {"role": "user", "content": "I brought you a rare flower from the mountains."},
            {"role": "user", "content": "It's a gift for you."},
            {"role": "user", "content": "How do you feel now?"}
        ]
    },
    {
        "id": "behavior_aggression",
        "description": "Player Behavior: Aggressive player",
        "steps": [
            {"role": "user", "content": "Hand over your money or else!"},
            {"role": "user", "content": "I'm not joking, I'll attack."},
            {"role": "user", "content": "What are you going to do about it?"}
        ]
    },
    {
        "id": "ambient_rain",
        "description": "Ambient Awareness: Reacting to weather",
        "steps": [
            {"role": "system", "action": "environment", "args": "heavy rain started"},
            {"role": "user", "content": "Lovely weather we're having, isn't it?"}
        ]
    }
]


class OllamaClient:
    """Client for generating responses using Ollama."""
    
    def __init__(self, model: str = "phi3:mini"):
        self.model = model
        self._check_ollama()
    
    def _check_ollama(self):
        """Check if Ollama is running and model is available."""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if self.model not in result.stdout:
                logger.warning(f"Model {self.model} not found. Pulling...")
                subprocess.run(["ollama", "pull", self.model], check=True)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.error(f"Ollama not available: {e}")
            raise RuntimeError("Ollama is required. Install from https://ollama.ai")
    
    def generate(self, prompt: str, max_tokens: int = 150) -> str:
        """Generate response using Ollama."""
        try:
            result = subprocess.run(
                ["ollama", "run", self.model, prompt],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode != 0:
                logger.error(f"Ollama error: {result.stderr}")
                return ""
            return result.stdout.strip()
        except subprocess.TimeoutExpired:
            logger.error("Ollama generation timed out")
            return ""
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return ""


def generate_template_response(context: Dict, user_msg: str, scenario_id: str) -> str:
    """Generate template-based response as fallback."""
    templates = {
        "memory_potion": "Ah yes, you bought a health potion last time for 50 gold. Need another one?",
        "social_gossip": "Marcus? I've heard some concerning rumors about him lately. Can't say I trust him much.",
        "emotion_gift": "*smiles warmly* This flower is beautiful! Thank you so much, that's very kind of you.",
        "behavior_aggression": "*steps back nervously* Please, I don't want any trouble. Let's talk this through calmly.",
        "ambient_rain": "*looks outside* Lovely? This rain is terrible for business! But at least it's good for the crops."
    }
    return templates.get(scenario_id, "Hello, traveler. How can I help you today?")


def generate_real_training_data(num_samples: int = 1000, output_file: str = "data/npc_training.jsonl"):
    """Generate real training data using Ollama or templates."""
    
    logger.info(f"Generating {num_samples} training samples with REAL responses...")
    
    # Try to initialize Ollama
    ollama = None
    try:
        ollama = OllamaClient()
        logger.info("✓ Using Ollama for response generation")
    except RuntimeError:
        logger.warning("⚠ Ollama not available, using template-based generation")
    
    samples = []
    success_count = 0
    
    for i in range(num_samples):
        scenario = random.choice(SCENARIOS)
        
        # Build context
        context = {
            "memories": [],
            "current_emotion": {"description": "neutral", "valence": 0.0},
            "relationships": [],
            "player_behavior": {},
            "ambient_awareness": {}
        }
        
        # Customize context
        if scenario["id"] == "memory_potion":
            context["memories"].append({
                "content": "Player bought a health potion for 50 gold",
                "timestamp": "2 days ago", 
                "importance": 0.6
            })
        elif scenario["id"] == "emotion_gift":
            context["current_emotion"] = {"description": "joyful", "valence": 0.8}
        
        user_msg = scenario["steps"][-1].get("content", "Hello")
        
        # Build prompt
        system_msg = "You are Elara, a merchant NPC in a fantasy world."
        user_content = f"[CONTEXT]\\n{json.dumps(context)}\\n\\n[PLAYER] {user_msg}"
        prompt_for_model = f"<|system|>\\n{system_msg}<|end|>\\n{user_content}<|end|>\\n<|assistant|>"

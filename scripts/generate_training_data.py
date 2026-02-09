#!/usr/bin/env python3
"""
BD-NSCA Training Data Generator

Generates synthetic training data for NPC dialogue fine-tuning.
Combines personas, scenarios, dynamic contexts, and player utterances
to create diverse prompt-response pairs.

Usage:
    python generate_training_data.py --output data/train.jsonl --samples 500
    python generate_training_data.py --output data/train.jsonl --samples 500 --use-llm
"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import logging

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent))

from colab_helpers import BDNSCAPromptFormatter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingSample:
    """A single training sample in BD-NSCA format."""
    id: str
    npc_id: str
    scenario_id: str
    persona: str
    plot: str
    context: Dict[str, str]
    player_input: str
    npc_response: str
    language: str
    
    def to_prompt_completion(self) -> Dict[str, str]:
        """Convert to prompt-completion format for fine-tuning."""
        prompt = BDNSCAPromptFormatter.format(
            persona=self.persona,
            plot=self.plot,
            context=self.context,
            player_input=self.player_input
        )
        return {
            "id": self.id,
            "prompt": prompt,
            "completion": self.npc_response,
            "metadata": {
                "npc_id": self.npc_id,
                "scenario_id": self.scenario_id,
                "language": self.language
            }
        }


class DataGenerator:
    """Generates training data by combining seed data components."""
    
    def __init__(self, data_dir: Path, language: str = "vi"):
        self.data_dir = data_dir
        self.language = language
        self.personas = self._load_json("personas.json")
        self.scenarios = self._load_json("scenarios.json")
        self.contexts = self._load_json("context_templates.json")
        self.utterances = self._load_json("player_utterances.json")
        
    def _load_json(self, filename: str) -> Dict:
        path = self.data_dir / filename
        if not path.exists():
            logger.warning(f"File not found: {path}")
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def _get_lang_key(self, obj: Dict, key_base: str) -> str:
        """Get language-specific key from object."""
        lang_key = f"{key_base}_{self.language}"
        if lang_key in obj:
            return obj[lang_key]
        # Fallback to nested dict with 'vi'/'en' keys
        if key_base in obj and isinstance(obj[key_base], dict):
            return obj[key_base].get(self.language, obj[key_base].get("vi", ""))
        return obj.get(key_base, "")
    
    def _build_context(self, scenario_id: str) -> Dict[str, str]:
        """Build a random dynamic context dictionary."""
        ctx = {}
        
        # Behavior state
        behavior = random.choice(list(self.contexts.get("behavior_states", {}).keys()))
        behavior_desc = self._get_lang_key(
            self.contexts["behavior_states"].get(behavior, {}), ""
        )
        if not behavior_desc and isinstance(self.contexts["behavior_states"].get(behavior), dict):
            behavior_desc = self.contexts["behavior_states"][behavior].get(self.language, behavior)
        ctx["Trạng thái hành vi" if self.language == "vi" else "Behavior State"] = behavior_desc or behavior
        
        # Health state
        health = random.choice(list(self.contexts.get("health_states", {}).keys()))
        health_data = self.contexts["health_states"].get(health, {})
        health_desc = health_data.get(self.language, health)
        ctx["Sức khỏe" if self.language == "vi" else "Health"] = health_desc
        
        # Mood state
        mood = random.choice(list(self.contexts.get("mood_states", {}).keys()))
        mood_desc = self.contexts["mood_states"].get(mood, {}).get(self.language, mood)
        ctx["Tâm trạng" if self.language == "vi" else "Mood"] = mood_desc
        
        # Nearby entities
        entities = random.choice(list(self.contexts.get("nearby_entities", {}).keys()))
        entities_desc = self.contexts["nearby_entities"].get(entities, {}).get(self.language, entities)
        ctx["Xung quanh" if self.language == "vi" else "Nearby"] = entities_desc
        
        # Time of day
        time_key = random.choice(list(self.contexts.get("time_contexts", {}).keys()))
        time_desc = self.contexts["time_contexts"].get(time_key, {}).get(self.language, time_key)
        ctx["Thời gian" if self.language == "vi" else "Time"] = time_desc
        
        return ctx
    
    def _generate_response_template(self, npc_id: str, context: Dict[str, str], player_input: str) -> str:
        """Generate a template NPC response based on context.
        
        This is a rule-based fallback. For better quality, use LLM generation.
        """
        npc = self.personas.get(npc_id, {})
        traits = npc.get("traits", [])
        
        # Get behavior and mood from context
        behavior_key = "Trạng thái hành vi" if self.language == "vi" else "Behavior State"
        mood_key = "Tâm trạng" if self.language == "vi" else "Mood"
        health_key = "Sức khỏe" if self.language == "vi" else "Health"
        
        behavior = context.get(behavior_key, "")
        mood = context.get(mood_key, "")
        health = context.get(health_key, "")
        
        # Template responses based on NPC and state
        responses_vi = {
            "gatekeeper": {
                "default": "Hmm. Đứng lại. Ngươi là ai?",
                "suspicious": "Ta không tin ngươi. Hãy chứng minh đi.",
                "alert": "Cảnh giác! Có kẻ lạ mặt!",
                "friendly": "Được rồi. Ngươi có vẻ đáng tin. Cứ vào đi.",
                "wounded": "Ugh... Ta bị thương nhưng vẫn phải canh gác."
            },
            "merchant": {
                "default": "À, khách quý! Mời xem hàng của tôi!",
                "suspicious": "Hmm, ngươi nhìn không có vẻ có tiền...",
                "friendly": "Chào bạn thân mến! Hôm nay tôi có nhiều hàng tốt lắm!",
                "working": "Đợi chút, tôi đang bận sắp xếp hàng...",
                "wounded": "Ôi, tôi bị thương trong chuyến đi vừa rồi..."
            },
            "healer": {
                "default": "Xin chào. Ngươi có cần thuốc men gì không?",
                "caring": "Ôi, trông ngươi có vẻ mệt mỏi. Để ta xem...",
                "friendly": "Chào mừng đến nhà thuốc của ta.",
                "working": "Ta đang sắc thuốc cho bệnh nhân khác, hãy đợi chút.",
                "wounded": "Ta cũng bị thương, nhưng vẫn có thể giúp ngươi."
            }
        }
        
        responses_en = {
            "gatekeeper": {
                "default": "Hmm. Halt. Who are you?",
                "suspicious": "I don't trust you. Prove yourself.",
                "alert": "Alert! Stranger approaching!",
                "friendly": "Alright. You seem trustworthy. Go ahead.",
                "wounded": "Ugh... I'm wounded but still on guard duty."
            },
            "merchant": {
                "default": "Ah, a customer! Come see my wares!",
                "suspicious": "Hmm, you don't look like you have coin...",
                "friendly": "Hello dear friend! I have great goods today!",
                "working": "Wait a moment, I'm arranging my goods...",
                "wounded": "Oh, I was hurt on my last journey..."
            },
            "healer": {
                "default": "Hello. Do you need any medicine?",
                "caring": "Oh, you look tired. Let me see...",
                "friendly": "Welcome to my medicine hut.",
                "working": "I'm brewing medicine for another patient, please wait.",
                "wounded": "I'm also wounded, but I can still help you."
            }
        }
        
        responses = responses_vi if self.language == "vi" else responses_en
        npc_responses = responses.get(npc_id, {"default": "..."})
        
        # Select response based on mood/state
        if "thương" in health.lower() or "wound" in health.lower() or "critical" in health.lower():
            return npc_responses.get("wounded", npc_responses["default"])
        if "nghi" in mood.lower() or "suspicious" in mood.lower():
            return npc_responses.get("suspicious", npc_responses["default"])
        if "thân thiện" in mood.lower() or "friendly" in mood.lower():
            return npc_responses.get("friendly", npc_responses["default"])
        if "alert" in behavior.lower() or "cảnh giác" in behavior.lower():
            return npc_responses.get("alert", npc_responses["default"])
        if "working" in behavior.lower() or "làm việc" in behavior.lower():
            return npc_responses.get("working", npc_responses["default"])
        
        return npc_responses["default"]
    
    def generate_sample(self, sample_id: int) -> TrainingSample:
        """Generate a single training sample."""
        # Select random NPC and scenario
        npc_id = random.choice(list(self.personas.keys()))
        scenario_id = random.choice(list(self.scenarios.keys()))
        
        npc = self.personas[npc_id]
        scenario = self.scenarios[scenario_id]
        
        # Get persona and plot
        persona = self._get_lang_key(npc, "persona")
        plot = self._get_lang_key(scenario, "plot")
        
        # Build dynamic context
        context = self._build_context(scenario_id)
        
        # Select random player utterance
        utterance_type = random.choice(list(self.utterances.keys()))
        utterances_list = self.utterances[utterance_type].get(self.language, [])
        if not utterances_list:
            utterances_list = ["Xin chào."] if self.language == "vi" else ["Hello."]
        player_input = random.choice(utterances_list)
        
        # Generate response (template-based or LLM)
        npc_response = self._generate_response_template(npc_id, context, player_input)
        
        return TrainingSample(
            id=f"sample_{sample_id:05d}",
            npc_id=npc_id,
            scenario_id=scenario_id,
            persona=persona,
            plot=plot,
            context=context,
            player_input=player_input,
            npc_response=npc_response,
            language=self.language
        )
    
    def generate_dataset(self, num_samples: int) -> List[TrainingSample]:
        """Generate multiple training samples."""
        samples = []
        for i in range(num_samples):
            samples.append(self.generate_sample(i))
        return samples


def generate_with_llm(samples: List[TrainingSample], model_id: str = "llama-3.1-8b") -> List[TrainingSample]:
    """Enhance sample responses using LLM (Teacher Model).
    
    Requires GROQ_API_KEY environment variable.
    """
    try:
        from colab_helpers import call_groq_api
    except ImportError:
        logger.warning("Could not import call_groq_api. Using template responses.")
        return samples
    
    import os
    if not os.environ.get("GROQ_API_KEY"):
        logger.warning("GROQ_API_KEY not set. Using template responses.")
        return samples
    
    logger.info(f"Enhancing {len(samples)} samples with LLM ({model_id})...")
    
    enhanced = []
    for sample in samples:
        # Create a meta-prompt for the teacher model
        meta_prompt = f"""You are helping create training data for an NPC dialogue system.

Given the following NPC persona, plot, and context, generate a single appropriate NPC response to the player's input. The response should:
1. Match the persona's personality and speaking style
2. Be appropriate for the current context (behavior state, mood, health)
3. Be in {'Vietnamese' if sample.language == 'vi' else 'English'}
4. Be 1-3 sentences long

Persona: {sample.persona}
Plot: {sample.plot}
Context: {json.dumps(sample.context, ensure_ascii=False)}
Player says: {sample.player_input}

NPC response:"""
        
        try:
            response = call_groq_api(meta_prompt, model_id=model_id, max_tokens=100, temperature=0.7)
            if response and len(response.strip()) > 0:
                sample.npc_response = response.strip()
        except Exception as e:
            logger.warning(f"LLM generation failed for sample {sample.id}: {e}")
        
        enhanced.append(sample)
    
    return enhanced


def save_jsonl(samples: List[TrainingSample], output_path: Path):
    """Save samples to JSONL format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            data = sample.to_prompt_completion()
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    
    logger.info(f"Saved {len(samples)} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate BD-NSCA training data")
    parser.add_argument("--output", "-o", default="data/train.jsonl", help="Output JSONL file")
    parser.add_argument("--samples", "-n", type=int, default=500, help="Number of samples to generate")
    parser.add_argument("--language", "-l", default="vi", choices=["vi", "en"], help="Language for generation")
    parser.add_argument("--use-llm", action="store_true", help="Use LLM to enhance responses")
    parser.add_argument("--llm-model", default="llama-3.3-70b-versatile", help="LLM model ID for enhancement")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data-dir", default=None, help="Directory containing seed data files")
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    # Determine data directory
    script_dir = Path(__file__).parent
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = script_dir.parent / "data"
    
    logger.info(f"Loading seed data from {data_dir}")
    generator = DataGenerator(data_dir, language=args.language)
    
    logger.info(f"Generating {args.samples} samples...")
    samples = generator.generate_dataset(args.samples)
    
    if args.use_llm:
        samples = generate_with_llm(samples, model_id=args.llm_model)
    
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = script_dir.parent / output_path
    
    save_jsonl(samples, output_path)
    
    # Print sample for verification
    if samples:
        logger.info("Sample output:")
        print(json.dumps(samples[0].to_prompt_completion(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

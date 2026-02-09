#!/usr/bin/env python3
"""
Generate Multi-Turn Training Data
Priority 3: Implementation of conversation history support.

Generates data in ChatML/Simulated Dialogue format:
System: Persona...
User: ...
Assistant: ...
User: ...
Assistant: ...
"""
import argparse
import json
import random
from pathlib import Path
from typing import List, Dict

def load_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def create_multiturn_prompt(persona: Dict, scenario: Dict, dialogue: List[Dict]) -> str:
    """Format a multi-turn conversation into a single training input."""
    # System prompt
    system_msg = f"System: {persona.get('persona_vi', persona.get('description', ''))}\nName: {persona['id']}\nContext: {scenario.get('name_vi', scenario.get('name', ''))}"
    
    # Conversation history
    conversation = []
    for turn in dialogue:
        role = turn["role"] # "user" or "assistant" (NPC)
        content = turn["content"]
        if role == "assistant":
            conversation.append(f"Answer: {content}") # Or "Assistant: "
        else:
            conversation.append(f"Question: {content}") # Or "User: "
            
    # Combine
    full_text = f"{system_msg}\n\n" + "\n".join(conversation)
    return full_text

import os
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from colab_helpers import call_groq_api
except ImportError:
    print("Warning: colab_helpers not found. LLM generation disabled.")
    call_groq_api = None

def generate_conversation_llm(persona: Dict, scenario: Dict, model: str = "llama-3.3-70b-versatile") -> List[Dict]:
    """Generate a multi-turn conversation using LLM."""
    if not call_groq_api or not os.environ.get("GROQ_API_KEY"):
        return []

    system_prompt = f"""You are a creative writer for an RPG game.
Generate a realistic dialogue between a Player (User) and an NPC (Assistant).

NPC Persona:
Name: {persona['name']} ({persona['id']})
Description: {persona['persona_vi']}
Traits: {', '.join(persona['traits'])}

Scenario:
Location: {scenario['name_vi']}
Context: {scenario['plot_vi']}

Requirements:
1. Generate 2-3 turns of conversation (User -> NPC -> User -> NPC).
2. The NPC must stay in character completely.
3. The Player should ask questions relevant to the scenario.
4. Language: Vietnamese.
5. Output MUST be valid JSON format: a list of objects with "role" ("user" or "assistant") and "content".
Example:
[
  {{"role": "user", "content": "..."}},
  {{"role": "assistant", "content": "..."}}
]
Do not output any text other than the JSON.
"""

    try:
        response = call_groq_api(system_prompt, model_id=model, temperature=0.8, max_tokens=1000)
        # Clean response to ensure JSON
        response = response.strip()
        if response.startswith("```json"):
            response = response.replace("```json", "").replace("```", "")
        
        dialogue = json.loads(response)
        return dialogue
    except Exception as e:
        print(f"Error generating conversation: {e}")
        return []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--personas", default="data/personas.json")
    parser.add_argument("--scenarios", default="data/scenarios.json")
    parser.add_argument("--output", default="data/train_multiturn.jsonl")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--model", default="llama-3.3-70b-versatile")
    args = parser.parse_args()
    
    # Load constraints
    try:
        personas = load_json(args.personas)
        scenarios = load_json(args.scenarios)
    except FileNotFoundError:
        print("Error: Seed files not found.")
        return

    print(f"Generating {args.samples} multi-turn samples using {args.model}...")
    
    samples = []
    generated_count = 0
    
    while generated_count < args.samples:
        npc_id = random.choice(list(personas.keys()))
        scenario_id = random.choice(list(scenarios.keys()))
        
        persona = personas[npc_id]
        scenario = scenarios[scenario_id] # scenarios is dict?
        # Check structure of scenarios.json
        # It's a dict: "village_gate": {...}
        
        # personas is dict: "gatekeeper": {...}
        
        dialogue = generate_conversation_llm(persona, scenario, model=args.model)
        
        if not dialogue:
            print("Skipping failed generation...")
            continue
            
        prompt = create_multiturn_prompt(persona, scenario, dialogue)
        
        # We want to train the model to continue the conversation or predict the last response.
        # Format: prompt = Full context + history. completion = last response?
        # Or standard CausalLM training on the whole text.
        
        samples.append({
            "prompt": prompt, 
            "completion": "", # Left empty for now, CausalLM uses "text" field usually
            "text": prompt,
            "metadata": {
                "npc_id": npc_id,
                "scenario_id": scenario_id,
                "turns": len(dialogue)
            }
        })
        
        # Write incrementally
        with open(args.output, "a", encoding="utf-8") as f:
            f.write(json.dumps(samples[-1], ensure_ascii=False) + "\n")
        
        generated_count += 1
        if generated_count % 5 == 0:
            print(f"Generated {generated_count}/{args.samples}")
            
    print(f"Saved {len(samples)} samples to {args.output}")

if __name__ == "__main__":
    main()

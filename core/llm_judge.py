"""
LLM-as-a-Judge Evaluation Script

Uses a strong LLM (OpenAI GPT-4o or Local Llama-3-70B via Ollama) to grade NPC responses.
Metrics:
1. Persona Consistency (1-5)
2. Hallucination Detection (Q² Method - Placeholder for future expansion)

Usage:
  python llm_judge.py --input responses.jsonl --model gpt-4o
"""

import argparse
import json
import os
import time
from typing import Dict, List, Optional
import requests

# OpenAI API (Set OPENAI_API_KEY env var)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

class LLMJudge:
    def __init__(self, provider: str = "ollama", model: str = "llama3:70b"):
        self.provider = provider
        self.model = model
        self.api_url = "http://localhost:11434/api/generate" if provider == "ollama" else "https://api.openai.com/v1/chat/completions"

    def evaluate_persona(self, persona: str, context: str, response: str) -> Dict:
        """Rate response consistency with persona (1-5)."""
        
        prompt = f"""
You are an expert roleplay critic. Rate the following NPC response based on the Persona.

**Persona:** {persona}
**Context:** {context}
**NPC Response:** {response}

**Criteria:**
1. Does the tone match the persona? (e.g. Grumpy vs Cheerful)
2. Is the language appropriate? (e.g. Medieval vs Modern)
3. Does it ignore game context?

Output strictly in JSON:
{{
  "score": <1-5 integer>,
  "reason": "<short justification>"
}}
"""
        
        result_text = self._call_llm(prompt)
        try:
            # Clean possible markdown code blocks
            clean_text = result_text.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_text)
            return data
        except json.JSONDecodeError:
            print(f"Error parse JSON from LLM: {result_text}")
            return {"score": 0, "reason": "Parse Error"}

    def _call_llm(self, prompt: str) -> str:
        if self.provider == "ollama":
            try:
                resp = requests.post(self.api_url, json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json" # Ollama supports json mode
                })
                resp.raise_for_status()
                return resp.json().get("response", "")
            except Exception as e:
                print(f"Ollama Error: {e}")
                return "{}"
        
        elif self.provider == "openai":
            if not OPENAI_API_KEY:
                return '{"score": 0, "reason": "Missing API Key"}'
            try:
                resp = requests.post(self.api_url, headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                }, json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"}
                })
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"OpenAI Error: {e}")
                return "{}"
        
        return "{}"

def main():
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge Eval")
    parser.add_argument("--provider", default="ollama", choices=["ollama", "openai"])
    parser.add_argument("--model", default="llama3:instruct")
    parser.add_argument("--input", required=True, help="Input JSONL file with {persona, context, response}")
    args = parser.parse_args()

    judge = LLMJudge(args.provider, args.model)
    
    results = []
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        # Create dummy file for demo if missing
        with open(args.input, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "persona": "You are a grumpy guard.",
                "context": "Player greets the guard.",
                "response": "Buzz off, rat!"
            }) + "\n")
            f.write(json.dumps({
                "persona": "You are a grumpy guard.",
                "context": "Player greets the guard.",
                "response": "Hello kind sir, welcome to our lovely village!"
            }) + "\n")
        print(f"Created dummy input file: {args.input}")

    print(f"Evaluating using {args.provider}/{args.model}...")
    
    with open(args.input, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    for line in lines:
        if not line.strip(): continue
        item = json.loads(line)
        
        eval_result = judge.evaluate_persona(item['persona'], item['context'], item['response'])
        
        print(f"\nResponse: {item['response']}")
        print(f"Score: {eval_result.get('score')}/5")
        print(f"Reason: {eval_result.get('reason')}")
        
        results.append({
            "response": item['response'],
            "score": eval_result.get('score'),
            "reason": eval_result.get('reason')
        })
        
    # Calculate average
    avg_score = sum(r['score'] for r in results if r['score']) / len(results) if results else 0
    print(f"\nAverage Persona Score: {avg_score:.2f}/5.0")

if __name__ == "__main__":
    main()

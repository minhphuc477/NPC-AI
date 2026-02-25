#!/usr/bin/env python3
"""
Backward-compatible entrypoint for NPC dialogue data collection.

This module now delegates to `collect_npc_dialogues_FIXED.py`, which
implements real response generation (Ollama with template fallback).
"""

from __future__ import annotations

import argparse

from collect_npc_dialogues_FIXED import OllamaClient, generate_real_training_data


def run_chat_interface(input_text: str, model: str = "phi3:mini") -> str:
    """
    Send one prompt to Ollama and return the generated text.
    Returns an empty string if Ollama is unavailable.
    """
    try:
        client = OllamaClient(model=model)
        return client.generate(input_text)
    except Exception:
        return ""


def generate_synthetic_data(num_samples: int = 100, output_file: str = "data/npc_training.jsonl") -> None:
    """
    Compatibility wrapper: generates real samples (or deterministic template fallback)
    via the fixed collector implementation.
    """
    generate_real_training_data(num_samples=num_samples, output_file=output_file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate NPC training dialogues.")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples to generate.")
    parser.add_argument("--output", type=str, default="data/npc_training.jsonl", help="Output JSONL file path.")
    args = parser.parse_args()

    generate_synthetic_data(num_samples=args.samples, output_file=args.output)


if __name__ == "__main__":
    main()

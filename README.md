# NPC AI - BD-NSCA Architecture

Vietnamese NPC dialogue generation using fine-tuned LLMs with the **Behavior-Driven Neuro-Symbolic Cognitive Architecture**.

## Quick Start

```bash
# 1. Generate training data
python scripts/generate_training_data.py --samples 500 --output data/train.jsonl

# 2. Train on Colab (upload notebooks/BD_NSCA_QLoRA_Finetune.ipynb)

# 3. Export to GGUF
python scripts/export_gguf.py --adapter outputs/adapter --output models/npc.gguf

# 4. Create Ollama model
ollama create npc-phi3 -f models/Modelfile.template

# 5. Run inference
python scripts/inference_adapter.py --test
```

## Project Structure

```
NPC AI/
├── data/                   # Seed data and training data
├── scripts/
│   ├── generate_training_data.py   # Data generation
│   ├── train_qlora.py              # QLoRA training
│   ├── inference_adapter.py        # Ollama inference
│   ├── export_gguf.py              # GGUF export
│   └── colab_helpers.py            # Prompt formatting + Groq API
├── evaluate/
│   ├── evaluate_bertscore.py       # BERTScore evaluation
│   └── QUALITATIVE_RUBRIC.md       # Human evaluation rubric
├── notebooks/
│   └── BD_NSCA_QLoRA_Finetune.ipynb  # Colab training notebook
└── models/
    └── Modelfile.template          # Ollama model template
```

## BD-NSCA Architecture

```
┌─────────────────┐
│ Symbolic Layer  │  UE5: AI Controller, Behavior Trees, Blackboard
└────────┬────────┘
         ▼
┌─────────────────┐
│ Middleware      │  Context Extractor → Prompt Formatter
└────────┬────────┘  [Persona] + [Plot] + [Context] + [Player Input]
         ▼
┌─────────────────┐
│ Neural Layer    │  Ollama + Fine-tuned Phi-3/Llama
└────────┬────────┘
         ▼
┌─────────────────┐
│ Feedback Layer  │  Dialogue Manager → Chat Bubbles
└─────────────────┘
```

## Requirements

- Python 3.10+
- PyTorch with CUDA (for training)
- Ollama (for inference)
- GROQ_API_KEY (for LLM-enhanced data generation)

## License

MIT License

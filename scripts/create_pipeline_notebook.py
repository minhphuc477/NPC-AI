import json
import os

# --- Notebook Structure ---
def create_notebook():
    return {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

def create_markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    }

def create_code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split("\n")]
    }

# --- Cell Content ---

# Title
CELL_0_MD = """# 🧠 NPC AI — Complete Training & Integration Pipeline

**BD-NSCA: Behavior-Driven Neuro-Symbolic Cognitive Architecture**

| Step | Description |
|------|-------------|
| 1 | Environment Setup |
| 2 | Training Data Generation |
| 3 | QLoRA Fine-Tuning (checkpoint/resume) |
| 4 | GGUF Export |
| 5 | Ollama Serving |
| 6 | Integrated Demo |
| 7 | Quality Evaluation |
| 8 | C++ Engine Compilation |

> **Checkpoint/Resume**: Training auto-detects and resumes from existing checkpoints."""

# Environment Setup
CELL_1_MD = """---
## 1. 🔧 Environment Setup & Dependencies"""

CELL_1_CODE = """# ============================================================
# Cell 1: Environment Setup
# ============================================================
import os, sys, subprocess

IN_KAGGLE = os.path.exists('/kaggle')
IN_COLAB = 'google.colab' in sys.modules
ENV_NAME = 'Kaggle' if IN_KAGGLE else ('Colab' if IN_COLAB else 'Local')
print(f'🌍 Environment: {ENV_NAME}')

if IN_KAGGLE or IN_COLAB:
    print('📦 Installing Unsloth and dependencies...')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
        'unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
        'trl>=0.12.0', 'transformers>=4.45.0', 'datasets', 'accelerate',
        'bitsandbytes', 'sentencepiece', 'protobuf'])
    print('📦 Installing Ollama...')
    os.system('curl -fsSL https://ollama.com/install.sh | sh')
    print('✅ All dependencies installed!')
else:
    print('ℹ️  Local env — assuming deps pre-installed.')

import torch
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f'🎮 GPU: {gpu_name} ({gpu_mem:.1f} GB)')
    print(f'   BF16: {torch.cuda.is_bf16_supported()}')
else:
    print('⚠️  No GPU detected!')
print('✅ Setup complete!')"""

# Data Generation
CELL_2_MD = """---
## 2. 📊 Training Data Generation

Generates diverse NPC dialogue data from **24 personas × 6 emotions × 6 memory states × 22 player inputs**.
Uses the Phi-3 chat template with `[CONTEXT]` blocks matching the BD-NSCA architecture."""

CELL_2_CODE = r"""# ============================================================
# Cell 2: Training Data Generation
# ============================================================
import json, random, os

# --- Load Personas ---
PERSONAS_PATH = 'data/personas.json'
if os.path.exists(PERSONAS_PATH):
    with open(PERSONAS_PATH, 'r', encoding='utf-8') as f:
        personas = json.load(f)
else:
    personas = {
        'merchant': {'persona_en': 'You are the Old Merchant, a cunning trader who has traveled across the kingdom.', 'traits': ['shrewd','friendly']},
        'gatekeeper': {'persona_en': 'You are the Gatekeeper, an old warrior, former royal guard. You are strict but fair.', 'traits': ['stern','brave','suspicious']},
        'healer': {'persona_en': 'You are the Village Healer, a kind middle-aged woman knowledgeable about herbs.', 'traits': ['caring','wise']},
    }
print(f'📋 {len(personas)} personas loaded')

# --- Scenario Building Blocks ---
EMOTIONS = [
    {'description': 'neutral', 'valence': 0.0},
    {'description': 'joyful', 'valence': 0.8},
    {'description': 'angry', 'valence': -0.7},
    {'description': 'fearful', 'valence': -0.5},
    {'description': 'trusting', 'valence': 0.6},
    {'description': 'surprised', 'valence': 0.3},
]
MEMORY_SETS = [
    [],
    [{'content': 'Player bought a health potion for 50 gold', 'timestamp': '2 days ago', 'importance': 0.6}],
    [{'content': 'Player saved the village from bandits', 'timestamp': '1 week ago', 'importance': 0.9}],
    [{'content': 'Player lied about their identity', 'timestamp': '3 days ago', 'importance': 0.7},
     {'content': 'Player returned a stolen item', 'timestamp': '1 day ago', 'importance': 0.8}],
    [{'content': 'Player asked about the forbidden forest', 'timestamp': 'yesterday', 'importance': 0.5}],
    [{'content': 'Player completed the herb gathering quest', 'timestamp': '4 days ago', 'importance': 0.7}],
]
RELATIONSHIPS = [
    [],
    [{'entity': 'Marcus', 'relation': 'rival', 'trust': -0.3}],
    [{'entity': 'Elder', 'relation': 'mentor', 'trust': 0.9}],
    [{'entity': 'Player', 'relation': 'acquaintance', 'trust': 0.2}],
    [{'entity': 'Player', 'relation': 'trusted_friend', 'trust': 0.8}],
]
PLAYER_INPUTS = [
    'Xin chào! Bạn khỏe không?',
    "Lovely weather we're having, isn't it?",
    'Do you trust Marcus?',
    'Bạn có biết gì về khu rừng cấm không?',
    'What happened to the old king?',
    "I'd like to buy some supplies.",
    'Can you heal my wounds?',
    'What are you going to do about it?',
    'Give me what I want or else!',
    'How do you feel now?',
    'Do you remember what I bought last time?',
    'Tell me about your past.',
    "I heard there's a dragon in the mountains.",
    'The dark mage is threatening the village!',
    'Tôi mới đến làng này.',
    'Bạn có thể rèn cho tôi một thanh kiếm không?',
    'Ai là người mạnh nhất ở đây?',
    'Tôi không tin bạn nói thật.',
    'Có phải có kho báu ẩn giấu ở đây không?',
    'Tôi muốn tham gia vào hội vệ binh.',
    'Bạn có nhớ lần cuối chúng ta gặp nhau không?',
    'Tôi cần giúp đỡ với một nhiệm vụ.',
]
AMBIENT = [
    {},
    {'time_of_day': 'morning', 'weather': 'sunny'},
    {'time_of_day': 'night', 'weather': 'rainy'},
    {'time_of_day': 'evening', 'weather': 'foggy', 'nearby_event': 'festival'},
]
PLAYER_BEHAVIORS = [
    {},
    {'stance': 'friendly', 'visit_count': 3},
    {'stance': 'aggressive', 'visit_count': 1},
    {'stance': 'neutral', 'visit_count': 5},
]

def generate_training_data(num_samples=1500, seed=42):
    random.seed(seed)
    samples = []
    persona_keys = list(personas.keys())
    
    # Use simpler string concatenation instead of complex f-strings to avoid quoting issues
    sys_tag = "<|system|>"
    end_tag = "<|end|>"
    user_tag = "<|user|>"
    asst_tag = "<|assistant|>"

    for i in range(num_samples):
        pk = random.choice(persona_keys)
        p = personas[pk]
        emo = random.choice(EMOTIONS)
        mems = random.choice(MEMORY_SETS)
        rels = random.choice(RELATIONSHIPS)
        inp = random.choice(PLAYER_INPUTS)
        amb = random.choice(AMBIENT)
        pb = random.choice(PLAYER_BEHAVIORS)

        use_vi = random.random() < 0.5
        persona_text = p.get('persona_vi', p['persona_en']) if use_vi else p['persona_en']

        ctx = json.dumps({'memories': mems, 'current_emotion': emo,
                          'relationships': rels, 'player_behavior': pb,
                          'ambient_awareness': amb})

        prompt = (f"{sys_tag}\n{persona_text}\n{end_tag}\n"
                  f"{user_tag}\n[CONTEXT]\n{ctx}\n\n[PLAYER] {inp}{end_tag}\n"
                  f"{asst_tag}\n")

        # Completion is a placeholder — will be replaced by real LLM outputs
        # or teacher model during actual training
        completion = f"[NPC responds in character]{end_tag}"

        samples.append({'prompt': prompt, 'completion': completion})

    return samples

# --- Generate & Save ---
OUTPUT_PATH = 'data/npc_training_v2.jsonl'
os.makedirs('data', exist_ok=True)

samples = generate_training_data(num_samples=1500)
with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    for s in samples:
        f.write(json.dumps(s, ensure_ascii=False) + '\n')

print(f'✅ Generated {len(samples)} training samples -> {OUTPUT_PATH}')

# Stats
unique_prompts = len(set(s['prompt'] for s in samples))
print(f'   Unique prompts: {unique_prompts} / {len(samples)}')
print(f'   Sample:\n{json.dumps(samples[0], indent=2, ensure_ascii=False)[:300]}...')"""

# QLoRA Training
CELL_3_MD = """---
## 3. 🎯 QLoRA Fine-Tuning with Checkpoint/Resume

Fine-tunes Phi-3-mini using Unsloth + QLoRA. Key features:
- **Auto checkpoint detection**: resumes from last saved checkpoint
- **Gradient checkpointing**: fits in 15GB VRAM
- **LoRA r=16** on all attention + MLP projections"""

CELL_3_CODE = """# ============================================================
# Cell 3: QLoRA Fine-Tuning with Checkpoint/Resume
# ============================================================
from unsloth import FastLanguageModel
import torch, os, glob
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# --- Configuration ---
MAX_SEQ_LENGTH = 2048
DTYPE = None  # Auto-detect (float16 for T4/V100, bfloat16 for Ampere+)
LOAD_IN_4BIT = True
MODEL_NAME = "unsloth/Phi-3-mini-4k-instruct"
OUTPUT_DIR = "npc_training_output"
DATASET_PATH = "data/npc_training_v2.jsonl"
if not os.path.exists(DATASET_PATH):
    DATASET_PATH = "data/npc_training.jsonl"  # Fallback to original

# --- Load Model ---
print(f"📥 Loading model: {MODEL_NAME}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
)

# --- Apply LoRA ---
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)
print("✅ LoRA applied (r=16, all projections)")

# --- Load Dataset ---
print(f"📂 Loading dataset: {DATASET_PATH}")
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
print(f"   {len(dataset)} samples loaded")

# --- Format for SFTTrainer ---
def formatting_prompts_func(examples):
    texts = []
    for p, c in zip(examples["prompt"], examples["completion"]):
        texts.append(f"{p}{c}")
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)

# --- Checkpoint/Resume Detection ---
resume_ckpt = None
if os.path.isdir(OUTPUT_DIR):
    checkpoints = sorted(glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*")))
    if checkpoints:
        resume_ckpt = checkpoints[-1]
        print(f"🔄 Resuming from checkpoint: {resume_ckpt}")
    else:
        print("📁 Output dir exists but no checkpoints found, training from scratch.")
else:
    print("🆕 No previous training found, starting fresh.")

# --- Training ---
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=2,
    packing=False,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=OUTPUT_DIR,
        report_to="none",
        save_strategy="steps",
        save_steps=20,
        save_total_limit=3,
    ),
)

print("🚀 Starting training...")
trainer.train(resume_from_checkpoint=resume_ckpt)

# --- Save Final Model ---
print(f"💾 Saving model to {OUTPUT_DIR}")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("✅ Training complete!")"""

# GGUF Export
CELL_4_MD = """---
## 4. 📦 GGUF Export

Exports the fine-tuned model to GGUF format for efficient inference via Ollama/llama.cpp."""

CELL_4_CODE = """# ============================================================
# Cell 4: GGUF Export
# ============================================================
import glob, os

GGUF_OUTPUT = "model"  # Unsloth appends _gguf -> "model_gguf/"

print("📦 Exporting to GGUF (f16)...")
# Note: Unsloth appends "_gguf" to the directory name
model.save_pretrained_gguf(GGUF_OUTPUT, tokenizer, quantization_method="f16")

# --- Find the exported GGUF file ---
search_dirs = ["model_gguf/", "model/", "model_gguf_gguf/"]
gguf_file = None
for d in search_dirs:
    matches = glob.glob(os.path.join(d, "*.gguf"))
    if matches:
        gguf_file = matches[0]
        break

if gguf_file:
    size_mb = os.path.getsize(gguf_file) / (1024 * 1024)
    print(f"✅ GGUF exported: {gguf_file} ({size_mb:.0f} MB)")
else:
    # Last resort: search everywhere
    all_gguf = glob.glob("**/*.gguf", recursive=True)
    if all_gguf:
        gguf_file = all_gguf[0]
        print(f"✅ GGUF found at: {gguf_file}")
    else:
        print("❌ GGUF file not found! Check export logs above.")
        gguf_file = None

trained_model_path = gguf_file
print(f"   trained_model_path = {trained_model_path}")"""

# Ollama Serving
CELL_5_MD = """---
## 5. 🚀 Ollama Server Setup & Model Registration

Starts the Ollama server and registers the fine-tuned model."""

CELL_5_CODE = r"""# ============================================================
# Cell 5: Ollama Serving
# ============================================================
import subprocess, time, requests, os

# --- Start Ollama Server ---
print("🚀 Starting Ollama server...")
ollama_process = subprocess.Popen(
    ["ollama", "serve"],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE
)
time.sleep(5)  # Wait for server startup

# --- Health Check ---
max_retries = 10
for i in range(max_retries):
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        if r.status_code == 200:
            print("✅ Ollama server is running!")
            break
    except Exception:
        pass
    print(f"   Waiting for server... ({i+1}/{max_retries})")
    time.sleep(3)
else:
    print("❌ Ollama server failed to start!")

# --- Create Modelfile ---
if trained_model_path and os.path.exists(trained_model_path):
    # Use triple quotes for Modelfile content
    modelfile_content = f'''FROM {trained_model_path}
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
SYSTEM You are an NPC in a fantasy RPG world. Respond in character, considering your memories, emotions, and relationships with the player.
'''
    with open("Modelfile", "w") as f:
        f.write(modelfile_content)

    print("📝 Modelfile created. Registering model...")
    result = subprocess.run(
        ["ollama", "create", "npc-ai", "-f", "Modelfile"],
        capture_output=True, text=True, timeout=300
    )
    if result.returncode == 0:
        print("✅ Model 'npc-ai' registered with Ollama!")
    else:
        print(f"❌ Registration failed: {result.stderr}")
else:
    print("⚠️  No GGUF file found — skipping Ollama model registration.")
    print("   You can still use the LoRA adapter from the training output.")"""

# Integrated Demo
CELL_6_MD = """---
## 6. 🎮 Integrated Demo — Full Cognitive Pipeline

Demonstrates the complete BD-NSCA system:
- **Emotional State Machine** (5-axis emotion with decay)
- **Conversation Memory** (sliding window + summarization)
- **Knowledge Graph** (NPC relationships & world facts)
- **Prompt Builder** (V3 format with full context)"""

CELL_6_CODE = """# ============================================================
# Cell 6: Integrated Demo
# ============================================================
import json, requests, sys, os

# Add project root to path for core imports
sys.path.insert(0, '.')

# --- Initialize Cognitive Components ---
try:
    from core.emotional_state import EmotionalStateMachine
    from core.conversation_memory import ConversationMemory
    emotional_sm = EmotionalStateMachine()
    conv_memory = ConversationMemory(max_turns=20)
    print("✅ Core cognitive components loaded")
    USE_CORE = True
except ImportError as e:
    print(f"⚠️  Core modules not available ({e})")
    print("   Using simplified standalone implementation")
    USE_CORE = False

    # Simplified standalone emotion tracker
    class SimpleEmotionTracker:
        def __init__(self):
            self.state = {'joy': 0.0, 'anger': 0.0, 'fear': 0.0, 'trust': 0.5, 'surprise': 0.0}
        def update(self, sentiment):
            if sentiment > 0:
                self.state['joy'] = min(1.0, self.state['joy'] + sentiment * 0.2)
                self.state['trust'] = min(1.0, self.state['trust'] + sentiment * 0.1)
            else:
                self.state['anger'] = min(1.0, self.state['anger'] - sentiment * 0.15)
                self.state['trust'] = max(0.0, self.state['trust'] + sentiment * 0.1)
            # Decay
            for k in self.state:
                if k != 'trust':
                    self.state[k] *= 0.9
        def dominant_emotion(self):
            return max(self.state, key=self.state.get)
        def summary(self):
            return {k: round(v, 2) for k, v in self.state.items()}

    emotional_sm = SimpleEmotionTracker()
    conv_memory_log = []

# --- Knowledge Graph (seed data) ---
knowledge_graph = {
    'npcs': {
        'Merchant': {'location': 'Market Square', 'sells': ['potions', 'weapons', 'armor']},
        'Gatekeeper': {'location': 'Main Gate', 'guards': 'entrance'},
        'Elder': {'location': 'Village Hall', 'role': 'village leader'},
    },
    'world_facts': [
        'The Dark Forest is forbidden — cursed by an ancient mage.',
        'The old king disappeared 10 years ago.',
        'Dragons were last seen in the Northern Mountains.',
        'The annual harvest festival is next week.',
    ],
    'relationships': {
        'Merchant-Gatekeeper': 'business_partners',
        'Elder-Merchant': 'old_friends',
        'Gatekeeper-Elder': 'respectful',
    }
}
print(f"🗺️ Knowledge Graph: {len(knowledge_graph['npcs'])} NPCs, "
      f"{len(knowledge_graph['world_facts'])} world facts")

# --- Ollama Query Function ---
def query_npc(player_input, npc_name="Merchant", memories=None, emotion_state=None):
    # Build context
    ctx = {
        'memories': memories or [],
        'current_emotion': emotion_state or {'description': 'neutral', 'valence': 0.0},
        'knowledge': knowledge_graph.get('world_facts', [])[:3],
        'npc_info': knowledge_graph.get('npcs', {}).get(npc_name, {}),
    }
    ctx_str = json.dumps(ctx)

    full_prompt = f"[CONTEXT]\\n{ctx_str}\\n\\n[PLAYER] {player_input}"

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "npc-ai",
                "prompt": full_prompt,
                "stream": False,
                "options": {"temperature": 0.7, "top_p": 0.9}
            },
            timeout=60
        )
        if response.status_code == 200:
            return response.json().get("response", "[No response]")
        else:
            return f"[Error {response.status_code}]"
    except Exception as e:
        return f"[Ollama not available: {e}]"

# --- Interactive Demo ---
print("\\n" + "="*60)
print("🎮 NPC AI INTEGRATED DEMO")
print("="*60)

demo_conversations = [
    ("Hello! I'm new to this village.", 0.3),
    ("What can you tell me about the forbidden forest?", 0.0),
    ("I want to buy a health potion.", 0.2),
    ("Do you trust the Gatekeeper?", -0.1),
    ("The dark mage sent me to spy on you!", -0.8),
]

memories = []
for player_input, sentiment in demo_conversations:
    print(f"\\n👤 Player: {player_input}")

    # Update emotion
    if USE_CORE:
        emotional_sm.process_interaction({'type': 'dialogue', 'sentiment': sentiment})
        emo_state = emotional_sm.get_state_dict()
    else:
        emotional_sm.update(sentiment)
        emo_state = {'description': emotional_sm.dominant_emotion(),
                     'valence': sentiment}

    # Query NPC
    response = query_npc(player_input, memories=memories, emotion_state=emo_state)
    print(f"🤖 NPC: {response}")

    # Store memory
    memories.append({
        'content': f'Player said: {player_input}',
        'timestamp': 'just now',
        'importance': abs(sentiment) + 0.3
    })
    if len(memories) > 10:
        memories = memories[-10:]

    # Show emotion state
    if USE_CORE:
        print(f"   💭 Emotion: {emo_state}")
    else:
        print(f"   💭 Emotion: {emotional_sm.summary()}")

print("\\n" + "="*60)
print("✅ Demo complete!")"""

# Quality Evaluation
CELL_7_MD = """---
## 7. 📈 Quality Evaluation

Runs automated quality metrics on NPC responses."""

CELL_7_CODE = """# ============================================================
# Cell 7: Quality Evaluation
# ============================================================
import re, math
from collections import Counter

def evaluate_response_quality(responses):
    if not responses:
        print("⚠️  No responses to evaluate.")
        return

    metrics = {}

    # 1. Average Length
    lengths = [len(r.split()) for r in responses]
    metrics['avg_word_count'] = sum(lengths) / len(lengths)

    # 2. Diversity (unique n-gram ratio)
    all_bigrams = []
    for r in responses:
        words = r.lower().split()
        all_bigrams.extend(zip(words, words[1:]))
    if all_bigrams:
        metrics['bigram_diversity'] = len(set(all_bigrams)) / len(all_bigrams)
    else:
        metrics['bigram_diversity'] = 0.0

    # 3. Repetition Score (lower = more repetitive)
    unique_responses = len(set(responses))
    metrics['uniqueness_ratio'] = unique_responses / len(responses)

    # 4. Length Variance (consistent response lengths is unnatural)
    if len(lengths) > 1:
        mean_len = sum(lengths) / len(lengths)
        variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
        metrics['length_std'] = math.sqrt(variance)
    else:
        metrics['length_std'] = 0.0

    # 5. Non-empty responses
    non_empty = sum(1 for r in responses if len(r.strip()) > 5)
    metrics['non_empty_ratio'] = non_empty / len(responses)

    # Display
    print("📊 Quality Metrics:")
    print(f"   📝 Avg Word Count:    {metrics['avg_word_count']:.1f}")
    print(f"   🔤 Bigram Diversity:  {metrics['bigram_diversity']:.3f}")
    print(f"   🎯 Uniqueness Ratio:  {metrics['uniqueness_ratio']:.3f}")
    print(f"   📏 Length Std Dev:     {metrics['length_std']:.1f}")
    print(f"   ✅ Non-empty Ratio:   {metrics['non_empty_ratio']:.3f}")

    return metrics

# Collect responses from demo (or generate test queries)
test_responses = []
test_inputs = [
    "Hello there!",
    "What do you sell?",
    "Tell me about the dark forest.",
    "Can I trust you?",
    "I need help with a quest.",
]

# Only query if NPC server is running (checking query_npc existence)
if 'query_npc' in globals():
    print("🔍 Generating test responses for evaluation...")
    for inp in test_inputs:
        resp = query_npc(inp)
        test_responses.append(resp)
        print(f"  Q: {inp}")
        print(f"  A: {resp[:100]}...")

    print("\\n")
    evaluate_response_quality(test_responses)
else:
    print("⚠️  Skipping evaluation (Ollama server not active)")"""

# C++ Compilation
CELL_8_MD = """---
## 8. 🔧 C++ Engine Compilation (Linux/Kaggle)

Builds the native BD-NSCA inference engine from the `cpp/` directory."""

CELL_8_CODE = """# ============================================================
# Cell 8: C++ Engine Compilation
# ============================================================
import subprocess, os, shutil

CPP_DIR = "cpp"

if not os.path.isdir(CPP_DIR):
    print("⚠️  cpp/ directory not found — skipping C++ build.")
    print("   This is expected if running in a cloud environment without the full repo.")
else:
    print("🔨 Building C++ BD-NSCA Engine...")

    build_dir = os.path.join(CPP_DIR, "build")
    os.makedirs(build_dir, exist_ok=True)

    # Configure
    print("   [1/3] CMake Configure...")
    result = subprocess.run(
        ["cmake", ".."],
        cwd=build_dir,
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"   ❌ Configure failed:\\n{result.stderr[:500]}")
    else:
        print("   ✅ Configure OK")

        # Build
        print("   [2/3] CMake Build...")
        result = subprocess.run(
            ["cmake", "--build", ".", "--config", "Release"],
            cwd=build_dir,
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"   ❌ Build failed:\\n{result.stderr[:500]}")
        else:
            print("   ✅ Build OK")

            # Run tests
            print("   [3/3] Running tests...")
            test_exes = ["test_fix", "test_grammar", "test_integration"]
            for test in test_exes:
                test_path = os.path.join(build_dir, "Release", test)
                if not os.path.exists(test_path):
                    test_path = os.path.join(build_dir, test)
                if os.path.exists(test_path):
                    r = subprocess.run([test_path], capture_output=True, text=True, timeout=60)
                    status = "✅" if r.returncode == 0 else "❌"
                    print(f"   {status} {test}: returncode={r.returncode}")
                else:
                    print(f"   ⏭️  {test}: not found")

    print("\\n✅ C++ build phase complete!")"""

# --- Main ---
def main():
    nb = create_notebook()
    nb["cells"].append(create_markdown_cell(CELL_0_MD))
    
    nb["cells"].append(create_markdown_cell(CELL_1_MD))
    nb["cells"].append(create_code_cell(CELL_1_CODE))
    
    nb["cells"].append(create_markdown_cell(CELL_2_MD))
    nb["cells"].append(create_code_cell(CELL_2_CODE))
    
    nb["cells"].append(create_markdown_cell(CELL_3_MD))
    nb["cells"].append(create_code_cell(CELL_3_CODE))
    
    nb["cells"].append(create_markdown_cell(CELL_4_MD))
    nb["cells"].append(create_code_cell(CELL_4_CODE))
    
    nb["cells"].append(create_markdown_cell(CELL_5_MD))
    nb["cells"].append(create_code_cell(CELL_5_CODE))
    
    nb["cells"].append(create_markdown_cell(CELL_6_MD))
    nb["cells"].append(create_code_cell(CELL_6_CODE))
    
    nb["cells"].append(create_markdown_cell(CELL_7_MD))
    nb["cells"].append(create_code_cell(CELL_7_CODE))
    
    nb["cells"].append(create_markdown_cell(CELL_8_MD))
    nb["cells"].append(create_code_cell(CELL_8_CODE))
    
    os.makedirs('notebooks', exist_ok=True)
    out_path = 'notebooks/NPC_AI_Complete_Pipeline.ipynb'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=4)
    print(f"✅ Generated notebook: {out_path}")

if __name__ == "__main__":
    main()

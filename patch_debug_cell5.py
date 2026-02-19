import json
import os

notebook_path = r'f:\NPC AI\notebooks\NPC_AI_Complete_Pipeline.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# --- Cell Sources ---

# Cell 1: Environment Setup
cell1_source = [
    "# ============================================================\n",
    "# Cell 1: Environment Setup (Auto-Clone & Install)\n",
    "# ============================================================\n",
    "import os, sys, subprocess, shutil\n",
    "\n",
    "IN_KAGGLE = os.path.exists('/kaggle')\n",
    "IN_COLAB = 'google.colab' in sys.modules\n",
    "ENV_NAME = 'Kaggle' if IN_KAGGLE else ('Colab' if IN_COLAB else 'Local')\n",
    "print(f'🌍 Environment: {ENV_NAME}')\n",
    "\n",
    "if IN_KAGGLE:\n",
    "    if not os.path.exists('NPC-AI'):\n",
    "        print('📥 Cloning NPC-AI repository...')\n",
    "        subprocess.run(['git', 'clone', 'https://github.com/minhphuc477/NPC-AI.git'], check=True)\n",
    "    \n",
    "    for folder in ['cpp', 'data']:\n",
    "        src = f'NPC-AI/{folder}'\n",
    "        if os.path.exists(src):\n",
    "            if not os.path.exists(folder):\n",
    "                print(f'📂 Cloning {folder} to root...')\n",
    "                shutil.copytree(src, folder)\n",
    "            else:\n",
    "                for item in os.listdir(src):\n",
    "                    s, d = os.path.join(src, item), os.path.join(folder, item)\n",
    "                    if not os.path.exists(d):\n",
    "                        if os.path.isdir(s): shutil.copytree(s, d)\n",
    "                        else: shutil.copy2(s, d)\n",
    "\n",
    "if IN_KAGGLE or IN_COLAB:\n",
    "    print('📦 Installing Unsloth and dependencies...')\n",
    "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git'])\n",
    "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', 'trl>=0.12.0', 'peft>=0.7.1', 'accelerate>=0.26.0', 'bitsandbytes>=0.40.0'])\n",
    "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'transformers>=4.45.0', 'datasets', 'sentencepiece', 'protobuf'])\n",
    "    \n",
    "    print('📦 Installing Ollama...')\n",
    "    try:\n",
    "        subprocess.run(['apt-get', 'update'], check=True, capture_output=True)\n",
    "        subprocess.run(['apt-get', 'install', '-y', 'zstd'], check=True, capture_output=True)\n",
    "        os.system('curl -fsSL https://ollama.com/install.sh | sh')\n",
    "        if shutil.which('ollama'): print('✅ Ollama installed successfully!')\n",
    "    except Exception as e: print(f'❌ Failed to install Ollama: {e}')\n",
    "\n",
    "import torch\n",
    "if torch.cuda.is_available():\n",
    "    print(f'🎮 GPU: {torch.cuda.get_device_name(0)}')\n",
    "else: print('⚠️  No GPU detected!')\n"
]

# Cell 2: Training Data Generation
cell2_source = [
    "# ============================================================\n",
    "# Cell 2: Training Data Generation (Strict English)\n",
    "# ============================================================\n",
    "import json, random, os\n",
    "os.makedirs('data', exist_ok=True)\n",
    "PERSONAS_PATH = 'data/personas.json'\n",
    "UTTERANCES_PATH = 'data/player_utterances.json'\n",
    "OUTPUT_PATH = 'data/npc_training_v2.json'\n",
    "\n",
    "if os.path.exists(PERSONAS_PATH):\n",
    "    with open(PERSONAS_PATH, 'r', encoding='utf-8') as f: personas = json.load(f)\n",
    "else: personas = {'merchant': {'persona_en': 'You are a Merchant.', 'traits': ['friendly'], 'id': 'merchant'}}\n",
    "\n",
    "if os.path.exists(UTTERANCES_PATH):\n",
    "    with open(UTTERANCES_PATH, 'r', encoding='utf-8') as f: utterances = json.load(f)\n",
    "else: utterances = {'greetings': {'en': ['Hello!']}}\n",
    "\n",
    "def generate_heuristic_response(persona, category, player_input):\n",
    "    name = persona.get('id', 'NPC').replace('npc_', '').capitalize()\n",
    "    traits = ', '.join(persona.get('traits', []))\n",
    "    if category == 'greetings': return f\"{name} nods. 'Greetings, traveler. I am the {name} with {traits} traits.'\"\n",
    "    elif category == 'trade_related': return f\"{name} eyes your gold. 'I have what you need, but the price is firm.'\"\n",
    "    return f\"{name} considers your words. 'I have much to think about regarding what you said.'\"\n",
    "\n",
    "dataset = []\n",
    "persona_list = list(personas.values())\n",
    "categories = list(utterances.keys())\n",
    "\n",
    "for _ in range(1500):\n",
    "    p = random.choice(persona_list)\n",
    "    c = random.choice(categories)\n",
    "    q = random.choice(utterances[c]['en'])\n",
    "    a = generate_heuristic_response(p, c, q)\n",
    "    \n",
    "    ctx = {'memories': [], 'current_emotion': {'description': 'neutral', 'valence': 0.0}, 'knowledge': [], 'npc_info': {'name': p.get('id', 'NPC'), 'persona': p.get('persona_en', '')}}\n",
    "    prompt = \"[INSTRUCTION] Respond strictly in English.\\n[CONTEXT]\\n\" + json.dumps(ctx, ensure_ascii=False) + \"\\n\\n[PLAYER] \" + q\n",
    "    dataset.append({'prompt': prompt, 'completion': a})\n",
    "\n",
    "with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:\n",
    "    json.dump(dataset, f, indent=1, ensure_ascii=False)\n",
    "print(f'✅ Generated {len(dataset)} STRICT ENGLISH training samples at {OUTPUT_PATH}')\n"
]

# Cell 3: Write Standalone Training Script
cell3_source = [
    "# ============================================================\n",
    "# Cell 3: Write Standalone Training Script\n",
    "# ============================================================\n",
    "import os\n",
    "os.makedirs('scripts', exist_ok=True)\n",
    "\n",
    "script_content = \"\"\"\n",
    "from unsloth import FastLanguageModel\n",
    "import torch\n",
    "from trl import SFTTrainer, SFTConfig\n",
    "from transformers import TrainingArguments\n",
    "from datasets import load_dataset\n",
    "import os\n",
    "import argparse\n",
    "\n",
    "def train(dataset_path, output_dir):\n",
    "    max_seq_length = 2048\n",
    "    dtype = None\n",
    "    load_in_4bit = True\n",
    "    model_name = \"unsloth/Phi-3-mini-4k-instruct\"\n",
    "    \n",
    "    print(f\"🚀 Loading Unsloth model: {model_name}\")\n",
    "    model, tokenizer = FastLanguageModel.from_pretrained(\n",
    "        model_name = model_name,\n",
    "        max_seq_length = max_seq_length,\n",
    "        dtype = dtype,\n",
    "        load_in_4bit = load_in_4bit,\n",
    "    )\n",
    "\n",
    "    model = FastLanguageModel.get_peft_model(\n",
    "        model, r = 16, target_modules = [\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\", \"gate_proj\", \"up_proj\", \"down_proj\",],\n",
    "        lora_alpha = 16, lora_dropout = 0, bias = \"none\", use_gradient_checkpointing = \"unsloth\",\n",
    "        random_state = 3407, use_rslora = False,\n",
    "    )\n",
    "\n",
    "    print(f\"📊 Loading dataset: {dataset_path}\")\n",
    "    dataset = load_dataset(\"json\", data_files=dataset_path, split=\"train\")\n",
    "    def formatting_prompts_func(examples):\n",
    "        texts = [f\"{p}{c}\" for p, c in zip(examples[\"prompt\"], examples[\"completion\"])]\n",
    "        return { \"text\" : texts, }\n",
    "    dataset = dataset.map(formatting_prompts_func, batched = True)\n",
    "\n",
    "    print(\"🚄 Starting training...\")\n",
    "    resume = os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0\n",
    "    trainer = SFTTrainer(\n",
    "        model = model, tokenizer = tokenizer, train_dataset = dataset, dataset_text_field = \"text\",\n",
    "        max_seq_length = max_seq_length, dataset_num_proc = 2, packing = False,\n",
    "        args = SFTConfig(\n",
    "            per_device_train_batch_size = 2, gradient_accumulation_steps = 4, warmup_steps = 5,\n",
    "            max_steps = 60, learning_rate = 2e-4, logging_steps = 1, optim = \"adamw_8bit\",\n",
    "            weight_decay = 0.01, seed = 3407, output_dir = output_dir, report_to = \"none\",\n",
    "            fp16 = not torch.cuda.is_bf16_supported(), bf16 = torch.cuda.is_bf16_supported(),\n",
    "        ),\n",
    "    )\n",
    "    trainer.train(resume_from_checkpoint = resume)\n",
    "    model.save_pretrained(output_dir)\n",
    "    tokenizer.save_pretrained(output_dir)\n",
    "    print(\"✅ Training complete!\")\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    parser = argparse.ArgumentParser()\n",
    "    parser.add_argument(\"--dataset\", type=str, required=True)\n",
    "    parser.add_argument(\"--output_dir\", type=str, default=\"outputs/npc_model\")\n",
    "    args = parser.parse_args()\n",
    "    train(args.dataset, args.output_dir)\n",
    "\"\"\"\n",
    "with open('scripts/train_unsloth.py', 'w') as f: f.write(script_content)\n",
    "print('✅ Standalone training script written to scripts/train_unsloth.py')\n"
]

# Cell 4: Execute Fine-tuning
cell4_source = [
    "# ============================================================\n",
    "# Cell 4: Execute Fine-tuning\n",
    "# ============================================================\n",
    "import subprocess, sys, os\n",
    "print('🚀 Starting fine-tuning...')\n",
    "subprocess.check_call([sys.executable, 'scripts/train_unsloth.py', \n",
    "                    '--dataset', 'data/npc_training_v2.json',\n",
    "                    '--output_dir', 'outputs/npc_model'])\n"
]

# Cell 5: Ollama Serving
cell5_source = [
    "# ============================================================\n",
    "# Cell 5: Ollama Serving (Robust Debug & English Only)\n",
    "# ============================================================\n",
    "import subprocess, time, requests, os, shutil, glob\n",
    "\n",
    "print(\"🚀 Check 1: Verifying Ollama binary...\")\n",
    "if not shutil.which('ollama'):\n",
    "    print(\"❌ Ollama binary not found! Cannot proceed.\")\n",
    "    ollama_process = None\n",
    "else:\n",
    "    print(\"🚀 Check 2: Starting Ollama server...\")\n",
    "    try:\n",
    "        if requests.get(\"http://localhost:11434/api/tags\", timeout=1).status_code == 200:\n",
    "            print(\"✅ Ollama is ALREADY running.\")\n",
    "            ollama_process = True\n",
    "        else: raise Exception(\"Not running\")\n",
    "    except:\n",
    "        ollama_process = subprocess.Popen([\"ollama\", \"serve\"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)\n",
    "        time.sleep(5)\n",
    "\n",
    "    server_ready = False\n",
    "    for i in range(12):\n",
    "        try:\n",
    "            if requests.get(\"http://localhost:11434/api/tags\", timeout=3).status_code == 200:\n",
    "                print(\"✅ Ollama server is running!\")\n",
    "                server_ready = True\n",
    "                break\n",
    "        except Exception: pass\n",
    "        print(f\"   Waiting for server... ({i+1}/12)\"); time.sleep(5)\n",
    "\n",
    "    if server_ready:\n",
    "        print(\"🚀 Check 3: Registering model...\")\n",
    "        tm_path = globals().get('trained_model_path')\n",
    "        if not tm_path or not os.path.exists(tm_path):\n",
    "            candidates = glob.glob(\"model_gguf/*.gguf\") + glob.glob(\"*.gguf\") + glob.glob(\"outputs/*.gguf\")\n",
    "            if candidates: tm_path = candidates[0]\n",
    "\n",
    "        if tm_path and os.path.exists(tm_path):\n",
    "            lines = [f'FROM {tm_path}', 'PARAMETER temperature 0.7', 'SYSTEM \"You are an NPC. Always respond strictly in English using English names. Do not use Vietnamese.\"']\n",
    "            modelfile = \"\\n\".join(lines)\n",
    "            with open(\"Modelfile\", \"w\") as f: f.write(modelfile)\n",
    "            print(f\"📦 Registering model 'npc-ai' from {tm_path}...\")\n",
    "            res = subprocess.run([\"ollama\", \"create\", \"npc-ai\", \"-f\", \"Modelfile\"], capture_output=True, text=True)\n",
    "            if res.returncode == 0: print(\"✅ Model registered successfully!\")\n",
    "            else: os.system(f\"ollama create npc-ai -f Modelfile\")\n",
    "        else: print(f\"❌ Model file NOT FOUND.\")\n",
    "    else: print(\"❌ Server never became ready.\")\n"
]

# Cell 6: Integrated Demo
cell6_source = [
    "# ============================================================\n",
    "# Cell 6: Integrated Demo (Strict English)\n",
    "# ============================================================\n",
    "import json, requests, os, time\n",
    "def query_npc(player_input, timeout=300):\n",
    "    ctx = {'memories': [], 'current_emotion': {'description': 'neutral', 'valence': 0.0}, 'knowledge': [], 'npc_info': {}}\n",
    "    prompt = \"[INSTRUCTION] Respond strictly in English.\\n[CONTEXT]\\n\" + json.dumps(ctx) + \"\\n\\n[PLAYER] \" + player_input\n",
    "    try:\n",
    "        res = requests.post(\"http://localhost:11434/api/generate\", json={\"model\": \"npc-ai\", \"prompt\": prompt, \"stream\": False}, timeout=timeout)\n",
    "        if res.status_code == 200: return res.json().get(\"response\", \"[No response]\")\n",
    "        return f\"[Error {res.status_code}]\"\n",
    "    except Exception as e: return f\"[Ollama error: {e}]\"\n",
    "\n",
    "print(\"🔍 Warming up model...\")\n",
    "print(\"Warmup:\", query_npc('Hello', timeout=300)[:20] + \"...\")\n",
    "for inp in [\"Hello! I am new here.\", \"What is the curse?\"]:\n",
    "    print(f\"\\n👤 Player: {inp}\\n🤖 NPC: {query_npc(inp)}\")\n"
]

# Cell 9: C++ Benchmarks
cell9_source = [
    "# ============================================================\n",
    "# Cell 9: C++ Engine Benchmarks\n",
    "# ============================================================\n",
    "import os, subprocess\n",
    "\n",
    "if os.path.exists('cpp/build'):\n",
    "    print(\"🚀 Running C++ Engine Benchmarks...\")\n",
    "    benchmarks = ['bench_engine', 'bench_memory', 'bench_retrieval', 'ablation_suite']\n",
    "    for bench in benchmarks:\n",
    "        path = f'cpp/build/{bench}'\n",
    "        if os.path.exists(path):\n",
    "            print(f\"\\n📊 Executing {bench}...\")\n",
    "            try:\n",
    "                res = subprocess.run([path], capture_output=True, text=True, timeout=300)\n",
    "                print(res.stdout)\n",
    "                if res.stderr: print(f\"⚠️ Stderr: {res.stderr}\")\n",
    "            except Exception as e: print(f\"❌ Failed to run {bench}: {e}\")\n",
    "        else: print(f\"⚠️ Benchmark binary not found: {path}\")\n",
    "else: print(\"❌ C++ build directory not found! Run Cell 8 first.\")\n"
]

# --- Patching Logic ---

patched = []

def replace_or_insert(target_marker, new_source, cell_id=None):
    global patched
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and any(target_marker in line for line in cell.get('source', [])):
            cell['source'] = new_source
            if cell_id: cell['id'] = cell_id
            patched.append(target_marker)
            return True
    return False

# 1. Patch existing cells
replace_or_insert('# Cell 1: Environment Setup', cell1_source)
replace_or_insert('# Cell 2: Training Data Generation', cell2_source)
replace_or_insert('scripts/train_unsloth.py', cell4_source) # The execute cell
replace_or_insert('# Cell 5: Ollama Serving', cell5_source)
replace_or_insert('# Cell 6: Integrated Demo', cell6_source)

# 2. Check for Cell 3 (Write script) - it might be missing
if not any('# Cell 3: Write Standalone Training Script' in str(c.get('source', [])) for c in nb['cells']):
    # Find position: after Cell 2
    for i, c in enumerate(nb['cells']):
        if any('# Cell 2' in str(line) for line in c.get('source', [])):
            nb['cells'].insert(i+1, {
                'cell_type': 'code',
                'id': 'write_script_cell',
                'metadata': {},
                'outputs': [],
                'source': cell3_source
            })
            patched.append('Cell 3 (Inserted)')
            break
else:
    replace_or_insert('# Cell 3: Write Standalone Training Script', cell3_source, 'write_script_cell')

# 3. Handle Cell 9
if not any('# Cell 9: C++ Engine Benchmarks' in str(c.get('source', [])) for c in nb['cells']):
    for i, c in enumerate(nb['cells']):
        if any('Compiling C++ engine' in str(line) for line in c.get('source', [])):
            nb['cells'].insert(i+1, {
                'cell_type': 'markdown',
                'id': 'bench_md',
                'metadata': {},
                'source': ['---\\n', '## 9. 📈 Benchmarking\\n']
            })
            nb['cells'].insert(i+2, {
                'cell_type': 'code',
                'id': 'bench_code',
                'metadata': {},
                'outputs': [],
                'source': cell9_source
            })
            patched.append('Cell 9 (Inserted)')
            break

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f'Successfully patched: {", ".join(patched)}')

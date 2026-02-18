import json
import os

notebook_path = r'f:\NPC AI\notebooks\NPC_AI_Complete_Pipeline.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Define cell sources as lists of strings
cell1_source = [
    "# ============================================================\n",
    "# Cell 1: Environment Setup\n",
    "# ============================================================\n",
    "import os, sys, subprocess, shutil\n",
    "\n",
    "IN_KAGGLE = os.path.exists('/kaggle')\n",
    "IN_COLAB = 'google.colab' in sys.modules\n",
    "ENV_NAME = 'Kaggle' if IN_KAGGLE else ('Colab' if IN_COLAB else 'Local')\n",
    "print(f'\\U0001f30d Environment: {ENV_NAME}')\n",
    "\n",
    "if IN_KAGGLE or IN_COLAB:\n",
    "    print('\\U0001f4e6 Installing Unsloth and dependencies...')\n",
    "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git'])\n",
    "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'trl>=0.12.0', 'transformers>=4.45.0', 'datasets', 'accelerate', 'bitsandbytes', 'sentencepiece', 'protobuf'])\n",
    "    print('\\U0001f4e6 Installing Ollama...')\n",
    "    try:\n",
    "        subprocess.run(['apt-get', 'update'], check=True, capture_output=True)\n",
    "        subprocess.run(['apt-get', 'install', '-y', 'zstd'], check=True, capture_output=True)\n",
    "        os.system('curl -fsSL https://ollama.com/install.sh | sh')\n",
    "        if shutil.which('ollama'): print('\\u2705 Ollama installed successfully!')\n",
    "    except Exception as e: print(f'\\u274c Failed to install Ollama: {e}')\n",
    "else: print('\\u2139\\ufe0f  Local env - assuming deps pre-installed.')\n",
    "\n",
    "import torch\n",
    "if torch.cuda.is_available():\n",
    "    print(f'\\U0001f3ae GPU: {torch.cuda.get_device_name(0)}')\n",
    "else: print('\\u26a0\\ufe0f  No GPU detected!')\n"
]

cell2_source = [
    "# ============================================================\n",
    "# Cell 2: Training Data Generation (Enhanced)\n",
    "# ============================================================\n",
    "import json, random, os\n",
    "PERSONAS_PATH = 'data/personas.json'\n",
    "UTTERANCES_PATH = 'data/player_utterances.json'\n",
    "if os.path.exists(PERSONAS_PATH):\n",
    "    with open(PERSONAS_PATH, 'r', encoding='utf-8') as f: personas = json.load(f)\n",
    "else: personas = {'merchant': {'persona_en': 'You are a Merchant.', 'traits': ['friendly'], 'name': 'Merchant'}}\n",
    "if os.path.exists(UTTERANCES_PATH):\n",
    "    with open(UTTERANCES_PATH, 'r', encoding='utf-8') as f: utterances = json.load(f)\n",
    "else: utterances = {'greetings': {'en': ['Hello!']}}\n",
    "def generate_heuristic_response(persona, category, player_input):\n",
    "    name = persona.get('name', 'NPC'); traits = \", \".join(persona.get('traits', []))\n",
    "    if category == 'greetings': return f\"{name} nods. 'Greetings, traveler. I am but a humble {name.lower()} with {traits} traits.'\"\n",
    "    elif category == 'trade_related': return f\"{name} eyes your gold. 'I have exactly what you need, but the price is firm.'\"\n",
    "    elif category == 'lore_questions': return f\"{name} looks distant. 'Ancient secrets are best left buried, though many whisper of the curse.'\"\n",
    "    return f\"{name} considers your words. 'I have much to think about regarding {player_input[:20]}...'\"\n",
    "def generate_training_data(num_samples=1500, seed=42):\n",
    "    random.seed(seed); samples = []\n",
    "    p_keys = list(personas.keys()); u_cats = list(utterances.keys())\n",
    "    for _ in range(num_samples):\n",
    "        pk = random.choice(p_keys); p = personas[pk]; cat = random.choice(u_cats)\n",
    "        inp = random.choice(utterances[cat].get('en', utterances[cat].get('vi', ['...'])))\n",
    "        ctx = json.dumps({'memories': [], 'current_emotion': {'description': 'neutral', 'valence': 0.0}})\n",
    "        prompt = f\"<|system|>\\n{p['persona_en']}\\n<|end|>\\n<|user|>\\n[CONTEXT]\\n{ctx}\\n\\n[PLAYER] {inp}<|end|>\\n<|assistant|>\\n\"\n",
    "        completion = f\"{generate_heuristic_response(p, cat, inp)}<|end|>\"\n",
    "        samples.append({'prompt': prompt, 'completion': completion})\n",
    "    return samples\n",
    "OUTPUT_PATH = 'data/npc_training_v2.jsonl'\n",
    "os.makedirs('data', exist_ok=True)\n",
    "samples = generate_training_data(num_samples=1500)\n",
    "with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:\n",
    "    for s in samples: f.write(json.dumps(s, ensure_ascii=False) + '\\n')\n",
    "print(f'\\u2705 Generated {len(samples)} training samples -> {OUTPUT_PATH}')\n"
]

cell5_source = [
    "# ============================================================\n",
    "# Cell 5: Ollama Serving\n",
    "# ============================================================\n",
    "import subprocess, time, requests, os, shutil\n",
    "if shutil.which('ollama'):\n",
    "    print(\"\\U0001f680 Starting Ollama server...\")\n",
    "    ollama_process = subprocess.Popen([\"ollama\", \"serve\"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)\n",
    "    time.sleep(5)\n",
    "else:\n",
    "    print(\"\\u274c Ollama binary not found!\"); ollama_process = None\n",
    "if ollama_process:\n",
    "    for i in range(12):\n",
    "        try:\n",
    "            if requests.get(\"http://localhost:11434/api/tags\", timeout=3).status_code == 200:\n",
    "                print(\"\\u2705 Ollama server is running!\"); break\n",
    "        except Exception: pass\n",
    "        print(f\"   Waiting for server... ({i+1}/12)\"); time.sleep(5)\n",
    "    if 'trained_model_path' in globals() and trained_model_path and os.path.exists(trained_model_path):\n",
    "        modelfile = f'FROM {trained_model_path}\\nPARAMETER temperature 0.7\\nSYSTEM You are an NPC.'\n",
    "        with open(\"Modelfile\", \"w\") as f: f.write(modelfile)\n",
    "        res = subprocess.run([\"ollama\", \"create\", \"npc-ai\", \"-f\", \"Modelfile\"], capture_output=True, text=True)\n",
    "        if res.returncode == 0: print(\"\\u2705 Model registered!\")\n",
    "        else: print(f\"\\u274c Registration failed: {res.stderr}\")\n"
]

cell6_source = [
    "# ============================================================\n",
    "# Cell 6: Integrated Demo (Enhanced)\n",
    "# ============================================================\n",
    "import json, requests, sys, os, time\n",
    "def query_npc(player_input, timeout=120):\n",
    "    ctx = {'memories': [], 'current_emotion': {'description': 'neutral', 'valence': 0.0}, 'knowledge': [], 'npc_info': {}}\n",
    "    prompt = f\"[CONTEXT]\\n{json.dumps(ctx)}\\n\\n[PLAYER] {player_input}\"\n",
    "    try:\n",
    "        res = requests.post(\"http://localhost:11434/api/generate\", json={\"model\": \"npc-ai\", \"prompt\": prompt, \"stream\": False, \"options\": {\"temperature\": 0.7}}, timeout=timeout)\n",
    "        if res.status_code == 200: return res.json().get(\"response\", \"[No response]\")\n",
    "        return f\"[Error {res.status_code}]\"\n",
    "    except Exception as e: return f\"[Ollama error: {e}]\"\n",
    "print(\"\\ud83d\\udd0d Warming up model...\")\n",
    "# No warmup query here, just define the function\n",
    "print(\"\\n\" + \"=\"*60 + \"\\n\\U0001f3ae NPC AI INTEGRATED DEMO\\n\" + \"=\"*60)\n",
    "for inp in [\"Hello! I am new here.\", \"What is the curse?\"]:\n",
    "    print(f\"\\n\\ud83d\\udc64 Player: {inp}\\n\\ud83e\\udd16 NPC: {query_npc(inp)}\")\n"
]

patched = 0
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        if any('# Cell 1: Environment Setup' in line for line in source):
            cell['source'] = cell1_source; patched += 1
        elif any('# Cell 2: Training Data Generation' in line for line in source):
            cell['source'] = cell2_source; patched += 1
        elif any('# Cell 5: Ollama Serving' in line for line in source):
            cell['source'] = cell5_source; patched += 1
        elif any('# Cell 6: Integrated Demo' in line for line in source):
            cell['source'] = cell6_source; patched += 1

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f'Successfully patched {patched} cells.')

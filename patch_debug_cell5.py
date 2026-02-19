import json
import os

notebook_path = r'f:\NPC AI\notebooks\NPC_AI_Complete_Pipeline.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Improved Cell 1: Environment Setup & Repo Clone
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
    "        if not os.path.exists(folder) and os.path.exists(f'NPC-AI/{folder}'):\n",
    "            print(f'📂 MOVING {folder} to root...')\n",
    "            if os.path.exists(folder): shutil.rmtree(folder)\n",
    "            shutil.copytree(f'NPC-AI/{folder}', folder)\n",
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

# Improved Cell 2: Training Data Generation (Strict English)
cell2_source = [
    "# ============================================================\n",
    "# Cell 2: Training Data Generation (Strict English)\n",
    "# ============================================================\n",
    "import json, random, os\n",
    "PERSONAS_PATH = 'data/personas.json'\n",
    "UTTERANCES_PATH = 'data/player_utterances.json'\n",
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
    "    dataset.append({'instruction': prompt, 'input': '', 'output': a})\n",
    "\n",
    "with open('dataset_npc.json', 'w', encoding='utf-8') as f:\n",
    "    json.dump(dataset, f, indent=1, ensure_ascii=False)\n",
    "print(f'✅ Generated {len(dataset)} STRICT ENGLISH training samples.')\n"
]

# Improved Cell 5: Debug prints, GGUF discovery, and Language Enforcement
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
    "        else:\n",
    "            raise Exception(\"Not running\")\n",
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

# Improved Cell 6: Integrated Demo (English Only)
cell6_source = [
    "# ============================================================\n",
    "# Cell 6: Integrated Demo (Strict English)\n",
    "# ============================================================\n",
    "import json, requests, os, time\n",
    "def query_npc(player_input, timeout=120):\n",
    "    ctx = {'memories': [], 'current_emotion': {'description': 'neutral', 'valence': 0.0}, 'knowledge': [], 'npc_info': {}}\n",
    "    prompt = \"[INSTRUCTION] Respond strictly in English.\\n[CONTEXT]\\n\" + json.dumps(ctx) + \"\\n\\n[PLAYER] \" + player_input\n",
    "    try:\n",
    "        res = requests.post(\"http://localhost:11434/api/generate\", json={\"model\": \"npc-ai\", \"prompt\": prompt, \"stream\": False}, timeout=timeout)\n",
    "        if res.status_code == 200: return res.json().get(\"response\", \"[No response]\")\n",
    "        return f\"[Error {res.status_code}]\"\n",
    "    except Exception as e: return f\"[Ollama error: {e}]\"\n",
    "\n",
    "print(\"🔍 Warming up model...\")\n",
    "print(\"Warmup:\", query_npc('Hello', timeout=180)[:20] + \"...\")\n",
    "for inp in [\"Hello! I am new here.\", \"What is the curse?\"]:\n",
    "    print(f\"\\n👤 Player: {inp}\\n🤖 NPC: {query_npc(inp)}\")\n"
]

# New Cell 9: C++ Benchmarks
cell9_source = [
    "# ============================================================\n",
    "# Cell 9: C++ Engine Benchmarks\n",
    "# ============================================================\n",
    "import os, subprocess\n",
    "\n",
    "if os.path.exists('cpp/build'):\n",
    "    print(\"🚀 Running C++ Engine Benchmarks...\")\n",
    "    benchmarks = ['bench_engine', 'bench_memory', 'bench_retrieval', 'ablation_suite']\n",
    "    \n",
    "    for bench in benchmarks:\n",
    "        path = f'cpp/build/{bench}'\n",
    "        if os.path.exists(path):\n",
    "            print(f\"\\n📊 Executing {bench}...\")\n",
    "            print('-'*40)\n",
    "            try:\n",
    "                res = subprocess.run([path], capture_output=True, text=True, timeout=300)\n",
    "                print(res.stdout)\n",
    "                if res.stderr: print(f\"⚠️ Stderr: {res.stderr}\")\n",
    "            except subprocess.TimeoutExpired:\n",
    "                print(f\"❌ {bench} timed out after 5 minutes.\")\n",
    "            except Exception as e:\n",
    "                print(f\"❌ Failed to run {bench}: {e}\")\n",
    "        else:\n",
    "            print(f\"⚠️ Benchmark binary not found: {path}\")\n",
    "else:\n",
    "    print(\"❌ C++ build directory (cpp/build) not found! Please run Cell 8 first.\")\n"
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

already_has_bench = any('# Cell 9: C++ Engine Benchmarks' in str(c.get('source', [])) for c in nb['cells'])

if not already_has_bench:
    pos = -1
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] == 'code' and any('Compiling C++ engine' in str(line) for line in c.get('source', [])):
            pos = i + 1
            break
    
    bench_cell = {
        'cell_type': 'code',
        'execution_count': None,
        'id': 'benchmarks_cell',
        'metadata': {},
        'outputs': [],
        'source': cell9_source
    }
    
    md_header = {
        'cell_type': 'markdown',
        'id': 'benchmarks_header',
        'metadata': {},
        'source': ['---\n', '## 9. 📈 Performance Benchmarking\n']
    }

    if pos != -1:
        nb['cells'] = nb['cells'][:pos] + [md_header, bench_cell] + nb['cells'][pos:]
    else:
        nb['cells'].append(md_header)
        nb['cells'].append(bench_cell)
    patched += 1

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f'Successfully patched {patched} cells.')

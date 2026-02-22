import json
import os

notebook_path = 'notebooks/NPC_AI_Complete_Pipeline.ipynb'
if not os.path.exists(notebook_path):
    print(f"❌ Notebook not found at {notebook_path}")
    exit(1)

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# --- Cell Sources Definition ---

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

# Cell 1.5: C++ Engine Patching (Robust Linux Build)
cell1_5_source = [
    "# ============================================================\n",
    "# Step 1.5: Patching C++ Engine for Linux compatibility\n",
    "# ============================================================\n",
    "import os, textwrap, re\n",
    "print('🛠️ Patching C++ engine code for Linux...')\n",
    "\n",
    "def apply_patch(filepath, old_str, new_str):\n",
    "    if not os.path.exists(filepath):\n",
    "        print(f'⚠️ File not found: {filepath}')\n",
    "        return\n",
    "    with open(filepath, 'r', encoding='utf-8') as f: content = f.read()\n",
    "    if old_str in content:\n",
    "        with open(filepath, 'w', encoding='utf-8') as f: f.write(content.replace(old_str, new_str))\n",
    "        print(f'✅ Patched {filepath}')\n",
    "    else: print(f'☑️ No patch needed for {filepath}')\n",
    "\n",
    "def prepend_to_file(filepath, prepend_str, guard=None):\n",
    "    if not os.path.exists(filepath):\n",
    "        print(f'⚠️ File not found: {filepath}')\n",
    "        return\n",
    "    with open(filepath, 'r', encoding='utf-8') as f: content = f.read()\n",
    "    if guard and guard in content:\n",
    "        print(f'☑️ Already has headers: {filepath}')\n",
    "        return\n",
    "    with open(filepath, 'w', encoding='utf-8') as f: f.write(prepend_str + content)\n",
    "    print(f'✅ Prepended to {filepath}')\n",
    "\n",
    "# Fix 1: PythonBridge.cpp — Linux process management headers\n",
    "prepend_to_file('cpp/src/PythonBridge.cpp', \n",
    "    '#include <thread>\\n#include <chrono>\\n#ifdef _WIN32\\n#include <windows.h>\\n#else\\n#include <unistd.h>\\n#include <sys/types.h>\\n#include <sys/wait.h>\\n#include <fcntl.h>\\n#include <signal.h>\\n#endif\\n', \n",
    "    guard='unistd.h')\n",
    "\n",
    "# Fix 2: NPCInference.cpp — missing thread/chrono\n",
    "prepend_to_file('cpp/src/NPCInference.cpp', '#include <thread>\\n#include <chrono>\\n', guard='<thread>')\n",
    "\n",
    "# Fix 3: HybridRetriever.h — signature fix\n",
    "apply_patch('cpp/include/HybridRetriever.h', \n",
    "    'Search(const std::string& query, const RetrievalConfig& config = {});', \n",
    "    'Search(const std::string& query);\\n    std::vector<RetrievalResult> Search(const std::string& query, const RetrievalConfig& config);')\n",
    "\n",
    "# Fix 4: PromptBuilder.cpp — Align with [NPC]/[PLAYER] format\n",
    "pb_new_source = textwrap.dedent(\"\"\"\n",
    "    std::string PromptBuilder::BuildAdvanced(const json& npcData, const json& gameState, const std::string& playerInput, const std::string& language, const json& tools) {\n",
    "        bool isVi = (language == \"vi\");\n",
    "        std::stringstream ss;\n",
    "        std::string persona = npcData.value(isVi ? \"persona_vi\" : \"persona_en\", npcData.value(\"persona\", isVi ? \"Bạn là một NPC.\" : \"You are an NPC.\"));\n",
    "        ss << \"[INSTRUCTION] \" << (isVi ? \"Trả lời bằng tiếng Việt. \" : \"Respond strictly in English. \") << persona << \"\\\\n\";\n",
    "        if (!tools.is_null() && !tools.empty()) ss << (isVi ? \"Công cụ khả dụng: \" : \"Available tools: \") << tools.dump() << \"\\\\n\";\n",
    "        ss << \"\\\\n[CONTEXT]\\\\n\";\n",
    "        json cog; cog[\"npc_info\"] = {{\"name\", npcData.value(\"name\", \"NPC\")}, {\"mood\", gameState.value(\"mood_state\", \"Neutral\")}, {\"health\", gameState.value(\"health_state\", \"Healthy\")}};\n",
    "        if (gameState.contains(\"current_emotion\")) cog[\"current_emotion\"] = gameState[\"current_emotion\"];\n",
    "        if (gameState.contains(\"memories\")) cog[\"memories\"] = gameState[\"memories\"];\n",
    "        if (gameState.contains(\"relationships\")) cog[\"relationships\"] = gameState[\"relationships\"];\n",
    "        if (gameState.contains(\"knowledge\")) cog[\"knowledge\"] = gameState[\"knowledge\"];\n",
    "        if (gameState.contains(\"recent_history\")) cog[\"recent_dialogue\"] = gameState[\"recent_history\"];\n",
    "        if (gameState.contains(\"memory_context\") && !gameState[\"memory_context\"].get<std::string>().empty()) cog[\"historical_memories\"] = gameState[\"memory_context\"];\n",
    "        ss << cog.dump() << \"\\\\n\\\\n[PLAYER] \" << playerInput << \"\\\\n[NPC] \";\n",
    "        return ss.str();\n",
    "    }\n",
    "\"\"\")\n",
    "with open('cpp/src/PromptBuilder.cpp', 'r', encoding='utf-8') as f: pb_content = f.read()\n",
    "pb_content = re.sub(r'std::string PromptBuilder::BuildAdvanced.*?\\n\\s+\\}', pb_new_source, pb_content, flags=re.DOTALL)\n",
    "with open('cpp/src/PromptBuilder.cpp', 'w', encoding='utf-8') as f: f.write(pb_content)\n",
    "print('✅ Overwrote PromptBuilder::BuildAdvanced')\n",
    "\n",
    "# Fix 5: NPCInference.cpp — Wire BuildAdvancedContext into Chat() with History\n",
    "chat_new_source = textwrap.dedent(\"\"\"\n",
    "    std::string NPCInferenceEngine::Chat(const std::string& session_id, const std::string& user_message) {\n",
    "        if (!conversation_manager_) return \"Error: No conversation manager\";\n",
    "        auto* ctx = conversation_manager_->GetSession(session_id);\n",
    "        if (!ctx) return \"Error: Invalid session ID\";\n",
    "        conversation_manager_->AddMessage(session_id, \"user\", user_message);\n",
    "        json advanced_context = BuildAdvancedContext(ctx->npc_name, user_message);\n",
    "        std::string history_str = \"\";\n",
    "        auto history = conversation_manager_->GetHistory(session_id, 6);\n",
    "        for (const auto& msg : history) history_str += (msg.role == \"user\" ? ctx->player_name : ctx->npc_name) + \": \" + msg.content + \"\\\\n\";\n",
    "        advanced_context[\"recent_history\"] = history_str;\n",
    "        advanced_context[\"npc_id\"] = ctx->npc_name;\n",
    "        advanced_context[\"player_id\"] = ctx->player_name;\n",
    "        advanced_context[\"conversation_id\"] = session_id;\n",
    "        std::string response = GenerateWithState(user_message, advanced_context, false);\n",
    "        conversation_manager_->AddMessage(session_id, \"assistant\", response);\n",
    "        if (config_.enable_graph) Learn(user_message);\n",
    "        return response;\n",
    "    }\n",
    "\"\"\")\n",
    "with open('cpp/src/NPCInference.cpp', 'r', encoding='utf-8') as f: ni_content = f.read()\n",
    "ni_content = re.sub(r'std::string NPCInferenceEngine::Chat.*?\\n\\s+\\}', chat_new_source, ni_content, flags=re.DOTALL)\n",
    "with open('cpp/src/NPCInference.cpp', 'w', encoding='utf-8') as f: f.write(ni_content)\n",
    "print('✅ Overwrote NPCInferenceEngine::Chat')\n",
    "\n",
    "print('🎉 C++ patching complete!')\n"
]

# Cell 2: Training Data Generation (Refined)
cell2_source = [
    "# ============================================================\n",
    "# Cell 2: Training Data Generation (Refined English)\n",
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
    "    traits = persona.get('traits', [])\n",
    "    trait_str = random.choice(traits) if traits else 'friendly'\n",
    "    templates = [\n",
    "        lambda: f\"{name} looks at you with a {trait_str} expression. 'Welcome, traveler. What brings you here?'\",\n",
    "        lambda: f\"'Ah, a new face!' {name} exclaims. 'I hope your journey was smoother than mine.'\",\n",
    "        lambda: f\"{name} pauses for a moment. 'I have much to share, but tell me, what is your business in this village?'\",\n",
    "        lambda: f\"The {name} nods slowly. 'Greetings. I am here to help, if you have the coin.'\"\n",
    "    ]\n",
    "    if category == 'greetings': return random.choice(templates)()\n",
    "    elif category == 'trade_related': return f\"{name} eyes your gear carefully. 'I deal in quality only. Are you buying or just looking?'\"\n",
    "    return f\"{name} considers your words deeply. 'Interesting... {player_input} is not something I hear every day.'\"\n",
    "\n",
    "dataset = []\n",
    "persona_list = list(personas.values())\n",
    "categories = list(utterances.keys())\n",
    "for _ in range(1200):\n",
    "    p = random.choice(persona_list)\n",
    "    c = random.choice(categories)\n",
    "    q = random.choice(utterances[c].get('en', ['Hello']))\n",
    "    a = generate_heuristic_response(p, c, q)\n",
    "    ctx = {'memories': [], 'current_emotion': {'description': 'neutral', 'valence': 0.0}, 'knowledge': [], 'npc_info': {'name': p.get('id', 'NPC'), 'persona': p.get('persona_en', '')}}\n",
    "    prompt = \"[INSTRUCTION] Respond strictly in English.\\n[CONTEXT]\\n\" + json.dumps(ctx, ensure_ascii=False) + \"\\n\\n[PLAYER] \" + q + \"\\n\\n[NPC] \"\n",
    "    dataset.append({'prompt': prompt, 'completion': a})\n",
    "\n",
    "with open(OUTPUT_PATH, 'w', encoding='utf-8') as f: json.dump(dataset, f, indent=1, ensure_ascii=False)\n",
    "print(f'✅ Generated {len(dataset)} REFINED training samples at {OUTPUT_PATH}')\n"
]

# Cell 3: Write Training Script
cell3_source = [
    "# ============================================================\n",
    "# Cell 3: Write Standalone Training Script (120 Steps)\n",
    "# ============================================================\n",
    "script_content = \"\"\"import torch, argparse, os\n",
    "from unsloth import FastLanguageModel\n",
    "from trl import SFTTrainer, SFTConfig\n",
    "from datasets import Dataset\n",
    "import json\n",
    "\n",
    "def train(dataset_path, output_dir):\n",
    "    model, tokenizer = FastLanguageModel.from_pretrained(\n",
    "        model_name = 'unsloth/Phi-3-mini-4k-instruct-bnb-4bit', max_seq_length = 2048, load_in_4bit = True\n",
    "    )\n",
    "    model = FastLanguageModel.get_peft_model(model, r = 16, target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj'], lora_alpha = 16, lora_dropout = 0, bias = 'none')\n",
    "    \n",
    "    with open(dataset_path, 'r') as f: data = json.load(f)\n",
    "    dataset = Dataset.from_list([{'text': d['prompt'] + d['completion'] + '<|end|>'} for d in data])\n",
    "\n",
    "    resume = os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0\n",
    "    trainer = SFTTrainer(\n",
    "        model = model, tokenizer = tokenizer, train_dataset = dataset, dataset_text_field = 'text',\n",
    "        max_seq_length = 2048, args = SFTConfig(\n",
    "            per_device_train_batch_size = 2, gradient_accumulation_steps = 4, warmup_steps = 10,\n",
    "            max_steps = 120, learning_rate = 2e-4, logging_steps = 1, optim = 'adamw_8bit',\n",
    "            output_dir = output_dir, report_to = 'none', fp16 = not torch.cuda.is_bf16_supported(), bf16 = torch.cuda.is_bf16_supported(),\n",
    "        ),\n",
    "    )\n",
    "    print('🚄 Training starting (120 steps)...')\n",
    "    trainer.train(resume_from_checkpoint = resume)\n",
    "    model.save_pretrained_merged(output_dir, tokenizer, save_method = 'merged_16bit')\n",
    "    print(f'✅ Model saved to {output_dir}')\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    parser = argparse.ArgumentParser()\n",
    "    parser.add_argument('--dataset', type=str, default='data/npc_training_v2.json')\n",
    "    parser.add_argument('--output_dir', type=str, default='outputs/npc_model')\n",
    "    args = parser.parse_args()\n",
    "    train(args.dataset, args.output_dir)\n",
    "\"\"\"\n",
    "with open('scripts/train_unsloth.py', 'w') as f: f.write(script_content)\n",
    "print('✅ Standalone training script written (120 steps)')\n"
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
    "# Cell 5: Ollama Serving (Safe Register & Stop Tokens)\n",
    "# ============================================================\n",
    "import subprocess, time, requests, os, glob\n",
    "print('🚀 Starting Ollama server...')\n",
    "try:\n",
    "    if requests.get('http://localhost:11434/api/tags', timeout=1).status_code == 200:\n",
    "        print('✅ Ollama is ALREADY running.')\n",
    "    else: raise Exception('Not running')\n",
    "except:\n",
    "    subprocess.Popen(['ollama', 'serve'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)\n",
    "    time.sleep(5)\n",
    "\n",
    "tm_path = globals().get('trained_model_path')\n",
    "if not tm_path or not os.path.exists(tm_path):\n",
    "    all_ggufs = glob.glob('model_gguf/*.gguf') + glob.glob('*.gguf') + glob.glob('outputs/*.gguf')\n",
    "    # Filter out vocab, embedding, and other non-model GGUFs\n",
    "    candidates = [f for f in all_ggufs if not any(x in f.lower() for x in ['vocab', 'embedding', 'bge', 'bert'])]\n",
    "    candidates = [f for f in candidates if os.path.exists(f) and os.path.getsize(f) > 500 * 1024 * 1024]\n",
    "    if candidates: tm_path = candidates[0]\n",
    "\n",
    "if tm_path and os.path.exists(tm_path):\n",
    "    lines = [\n",
    "        f'FROM {tm_path}', 'PARAMETER temperature 0.7',\n",
    "        'PARAMETER stop \"[PLAYER]\"', 'PARAMETER stop \"[INSTRUCTION]\"', 'PARAMETER stop \"[CONTEXT]\"', 'PARAMETER stop \"<|end|>\"',\n",
    "        'SYSTEM \"You are an NPC. Always respond strictly in English as the [NPC] speaker. Do not repeat the prompt.\"'\n",
    "    ]\n",
    "    with open('Modelfile', 'w') as f: f.write('\\n'.join(lines))\n",
    "    print(f'📦 Registering model npc-ai from {tm_path}...')\n",
    "    subprocess.run(['ollama', 'create', 'npc-ai', '-f', 'Modelfile'])\n",
    "else: print('❌ Model file NOT FOUND.')\n"
]

# Cell 6: Demo
cell6_source = [
    "# ============================================================\n",
    "# Cell 6: Integrated Demo (Clean Turns)\n",
    "# ============================================================\n",
    "import json, requests\n",
    "def query_npc(player_input):\n",
    "    ctx = {'memories': [], 'current_emotion': {'description': 'neutral', 'valence': 0.0}, 'npc_info': {'name': 'Blacksmith', 'persona': 'A friendly blacksmith.'}}\n",
    "    prompt = \"[INSTRUCTION] Respond strictly in English.\\n[CONTEXT]\\n\" + json.dumps(ctx) + \"\\n\\n[PLAYER] \" + player_input + \"\\n\\n[NPC] \"\n",
    "    try:\n",
    "        payload = {\"model\": \"npc-ai\", \"prompt\": prompt, \"stream\": False, \"options\": {\"stop\": [\"[PLAYER]\", \"[INSTRUCTION]\", \"<|end|>\"]}}\n",
    "        res = requests.post(\"http://localhost:11434/api/generate\", json=payload, timeout=60)\n",
    "        if res.status_code == 200:\n",
    "            text = res.json().get('response', '[No response]')\n",
    "            return text.split('[NPC]')[-1].strip()\n",
    "        return f\"[Error {res.status_code}]\"\n",
    "    except Exception as e: return f\"[Error: {e}]\"\n",
    "\n",
    "for inp in [\"Hello! I am new here.\", \"What is the curse?\"]:\n",
    "    print(f\"👤 Player: {inp}\\n🤖 NPC: {query_npc(inp)}\\n\")\n"
]

# Cell 8: Compilation
cell8_source = [
    "# ============================================================\n",
    "# Cell 8: C++ Engine Compilation (Optimized)\n",
    "# ============================================================\n",
    "import os, subprocess\n",
    "if os.path.exists('cpp'):\n",
    "    os.makedirs('cpp/build', exist_ok=True)\n",
    "    try:\n",
    "        subprocess.check_call(['cmake', '..'], cwd='cpp/build')\n",
    "        nproc = subprocess.check_output(['nproc']).decode().strip()\n",
    "        subprocess.check_call(['make', f'-j{nproc}'], cwd='cpp/build')\n",
    "        print('✅ Compilation successful!')\n",
    "    except subprocess.CalledProcessError as e: print(f'❌ Failed: {e}')\n",
    "else: print('⚠️ cpp/ not found.')\n"
]

# Cell 9: Benchmarks
cell9_source = [
    "# Cell 9: Benchmarks\n",
    "import subprocess, os\n",
    "for b in ['bench_engine', 'bench_memory', 'bench_retrieval', 'ablation_suite']:\n",
    "    p = f'cpp/build/{b}'\n",
    "    if os.path.exists(p):\n",
    "        print(f'📊 Running {b}...')\n",
    "        subprocess.run([p])\n"
]

# --- Patching Logic ---

def replace_or_insert(nb, target_marker, new_source, cell_id=None):
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and any(target_marker in str(line) for line in cell.get('source', [])):
            cell['source'] = new_source
            if cell_id: cell['id'] = cell_id
            return True
    return False

# 1. Patch Core Cells
replace_or_insert(nb, '# Cell 1: Environment Setup', cell1_source)
replace_or_insert(nb, 'Step 1.5: Patching C++ Engine', cell1_5_source, 'cpp_patch_cell')
replace_or_insert(nb, '# Cell 2: Training Data Generation', cell2_source)
replace_or_insert(nb, 'scripts/train_unsloth.py', cell4_source)
replace_or_insert(nb, '# Cell 5: Ollama Serving', cell5_source)
replace_or_insert(nb, '# Cell 6: Integrated Demo', cell6_source)
replace_or_insert(nb, 'Compiling C++ engine', cell8_source)

# 2. Insert missing ones (Cell 1.5, Cell 3, Cell 9 if needed)
if not any('Step 1.5' in str(c.get('source', [])) for c in nb['cells']):
    for i, c in enumerate(nb['cells']):
        if '# Cell 1' in str(c.get('source', [])): 
            nb['cells'].insert(i+1, {'cell_type': 'code', 'id': 'cpp_patch_cell', 'metadata': {}, 'outputs': [], 'source': cell1_5_source})
            break

if not any('# Cell 3' in str(c.get('source', [])) for c in nb['cells']):
    for i, c in enumerate(nb['cells']):
        if '# Cell 2' in str(c.get('source', [])): 
            nb['cells'].insert(i+1, {'cell_type': 'code', 'id': 'write_train', 'metadata': {}, 'outputs': [], 'source': cell3_source})
            break

if not any('Cell 9' in str(c.get('source', [])) for c in nb['cells']):
    nb['cells'].append({'cell_type': 'code', 'id': 'bench_cell', 'metadata': {}, 'outputs': [], 'source': cell9_source})

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("✅ Master notebook updated with C++ fixes and NPC quality improvements.")

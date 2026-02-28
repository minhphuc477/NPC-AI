#!/usr/bin/env python3
# ======================================================================
# Auto-generated from NPC_AI_Complete_Pipeline.ipynb
# ======================================================================


# ======================================================================
# MARKDOWN CELL 1
# ======================================================================
# # 🧠 NPC AI — Complete Training & Integration Pipeline
# 
# **BD-NSCA: Behavior-Driven Neuro-Symbolic Cognitive Architecture**
# 
# | Step | Description |
# |------|-------------|
# | 1 | Environment Setup |
# | 2 | Training Data Generation |
# | 3 | QLoRA Fine-Tuning (checkpoint/resume) |
# | 4 | GGUF Export |
# | 5 | Ollama Serving |
# | 6 | Integrated Demo |
# | 7 | Quality Evaluation |
# | 8 | C++ Engine Compilation |
# 
# > **Checkpoint/Resume**: Training auto-detects and resumes from existing checkpoints.



# ======================================================================
# MARKDOWN CELL 2
# ======================================================================
# ---
# ## 1. 🔧 Environment Setup & Dependencies



# ======================================================================
# CODE CELL 3
# ======================================================================
# ============================================================
# Cell 1: Environment Setup (accelerator-aware: TPU/GPU/CPU)
# ============================================================
import os
import subprocess
import sys
from pathlib import Path

import shutil

KAGGLE_DATASET_CODE_PATHS = [
    Path('/kaggle/input/datasets/mphacc3/npc-ai'),
    Path('/kaggle/input/npc-ai'),
]


def _has_repo_scripts(root: Path) -> bool:
    return (root / 'scripts' / 'run_kaggle_full_results.py').exists()


def _resolve_dataset_repo_root() -> Path | None:
    for base in KAGGLE_DATASET_CODE_PATHS:
        for candidate in (base, base / 'npc-ai', base / 'NPC AI', base / 'NPC_AI'):
            if _has_repo_scripts(candidate):
                return candidate
    return None


def bootstrap_repo_root_for_kaggle() -> Path:
    cwd = Path.cwd()
    if _has_repo_scripts(cwd):
        return cwd

    if not Path('/kaggle').exists():
        print(f'Warning: expected repo files not found in {cwd}')
        return cwd

    working_repo = Path('/kaggle/working/npc-ai')
    if _has_repo_scripts(working_repo):
        os.chdir(working_repo)
        print(f'Using existing working repo: {working_repo}')
        return working_repo

    source_repo = _resolve_dataset_repo_root()
    if source_repo is None:
        print('Warning: Kaggle dataset repo path not found; continuing in current directory.')
        return cwd

    shutil.copytree(source_repo, working_repo, dirs_exist_ok=True)
    os.chdir(working_repo)
    print(f'Copied repo from {source_repo} -> {working_repo}')
    return working_repo


REPO_ROOT = bootstrap_repo_root_for_kaggle()


IN_KAGGLE = Path('/kaggle').exists()
IN_COLAB = 'google.colab' in sys.modules
ENV_NAME = 'Kaggle' if IN_KAGGLE else ('Colab' if IN_COLAB else 'Local')


def run_cmd(cmd, allow_fail=False):
    try:
        subprocess.check_call(cmd)
        return True
    except Exception as exc:
        if not allow_fail:
            raise
        print(f'Warning: command failed: {cmd}')
        print(f'  -> {exc}')
        return False


def pip_install(packages, allow_fail=False, extra_args=None):
    extra_args = extra_args or []
    cmd = [sys.executable, '-m', 'pip', 'install', '-q'] + list(extra_args) + list(packages)
    return run_cmd(cmd, allow_fail=allow_fail)


def detect_runtime():
    forced = os.environ.get('NPC_ACCELERATOR', 'auto').strip().lower()
    if forced in {'tpu', 'cuda', 'cpu'}:
        return forced

    if os.environ.get('PJRT_DEVICE', '').strip().upper() == 'TPU':
        return 'tpu'

    kaggle_accel = os.environ.get('KAGGLE_ACCELERATOR_TYPE', '').strip().upper()
    if kaggle_accel.startswith('TPU'):
        return 'tpu'

    tpu_hints = ('TPU_NAME', 'COLAB_TPU_ADDR', 'TPU_WORKER_ID')
    if any(os.environ.get(k) for k in tpu_hints):
        try:
            import torch_xla.core.xla_model as xm  # type: ignore
            _ = xm.xla_device()
            return 'tpu'
        except Exception:
            pass

    try:
        import torch_xla.core.xla_model as xm  # type: ignore
        dev = str(xm.xla_device()).lower()
        if 'xla' in dev:
            return 'tpu'
    except Exception:
        pass

    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
    except Exception:
        pass

    return 'cpu'


TRAIN_ACCELERATOR = detect_runtime()
if TRAIN_ACCELERATOR == 'tpu':
    os.environ.setdefault('PJRT_DEVICE', 'TPU')

print(f'Environment: {ENV_NAME}')
print(f'Training accelerator: {TRAIN_ACCELERATOR}')

base_deps = [
    'transformers>=4.45.0',
    'datasets>=2.20.0',
    'peft>=0.11.0',
    'accelerate>=0.30.0',
    'sentencepiece',
    'protobuf',
    'requests',
]
pip_install(base_deps, allow_fail=False)

if TRAIN_ACCELERATOR == 'cuda':
    pip_install(['bitsandbytes>=0.43.0'], allow_fail=True)
    # Optional, only for legacy Unsloth path.
    pip_install(['unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git'], allow_fail=True)
elif TRAIN_ACCELERATOR == 'tpu':
    try:
        import torch_xla  # type: ignore # noqa: F401
        print('torch_xla already available.')
    except Exception:
        print('torch_xla not found. Attempting install for TPU runtime...')
        pip_install(
            ['torch_xla[tpu]>=2.2'],
            allow_fail=True,
            extra_args=['-f', 'https://storage.googleapis.com/libtpu-releases/index.html'],
        )

try:
    import torch
    if torch.cuda.is_available():
        print(f'CUDA GPU count: {torch.cuda.device_count()}')
        print(f'Primary GPU: {torch.cuda.get_device_name(0)}')
except Exception as exc:
    print(f'Warning: torch check failed: {exc}')

if TRAIN_ACCELERATOR == 'tpu':
    try:
        import torch_xla.core.xla_model as xm  # type: ignore
        print(f'TPU device: {xm.xla_device()}')
    except Exception as exc:
        print(f'Warning: TPU selected but torch_xla check failed: {exc}')

print('Environment setup done.')



# ======================================================================
# CODE CELL 4
# ======================================================================
# ============================================================
# Step 1.5: Validate C++ Engine Layout (Non-destructive)
# ============================================================
import os
from pathlib import Path

print('Validating C++ engine files for Kaggle build...')

required_files = [
    'cpp/CMakeLists.txt',
    'cpp/src/NPCInference.cpp',
    'cpp/src/PromptBuilder.cpp',
    'cpp/include/ModelLoader.h',
]
missing = [p for p in required_files if not os.path.exists(p)]

if missing:
    print('Warning: missing required C++ files:')
    for m in missing:
        print(f'  - {m}')
else:
    print('All required C++ files are present.')

pb_path = Path('cpp/src/PromptBuilder.cpp')
if pb_path.exists():
    pb_text = pb_path.read_text(encoding='utf-8', errors='ignore')
    if 'BuildAdvanced' in pb_text and '[CONTEXT]' in pb_text and '[PLAYER]' in pb_text:
        print('PromptBuilder format check passed.')
    else:
        print('Warning: PromptBuilder format markers were not detected.')
else:
    print('Warning: PromptBuilder.cpp not found.')

print('C++ validation complete (source files left unchanged).')



# ======================================================================
# MARKDOWN CELL 5
# ======================================================================
# ---
# ## 2. 📝 Training Data Generation (Enhanced)



# ======================================================================
# CODE CELL 6
# ======================================================================
# ============================================================
# Cell 2: Training Data Generation (Refined English)
# ============================================================
import json
import os
import random

os.makedirs('data', exist_ok=True)
PERSONAS_PATH = 'data/personas.json'
UTTERANCES_PATH = 'data/player_utterances.json'
OUTPUT_PATH = 'data/npc_training_v2.json'

if os.path.exists(PERSONAS_PATH):
    with open(PERSONAS_PATH, 'r', encoding='utf-8') as f:
        personas = json.load(f)
else:
    personas = {'merchant': {'persona_en': 'You are a Merchant.', 'traits': ['friendly'], 'id': 'merchant'}}

if os.path.exists(UTTERANCES_PATH):
    with open(UTTERANCES_PATH, 'r', encoding='utf-8') as f:
        utterances = json.load(f)
else:
    utterances = {'greetings': {'en': ['Hello!']}}


import sys
sys.path.insert(0, str(Path(__file__).parent))
try:
    from scripts.colab_helpers import call_groq_api
except ImportError:
    call_groq_api = None
    print("Warning: colab_helpers not found. Falling back to heuristic generation.")

def generate_heuristic_response(persona, category, player_input):
    name = persona.get('id', 'NPC').replace('npc_', '').capitalize()
    traits = persona.get('traits', [])
    trait_str = random.choice(traits) if traits else 'friendly'
    templates = [
        lambda: f"{name} looks at you with a {trait_str} expression. 'Welcome, traveler. What brings you here?'",
        lambda: f"'Ah, a new face!' {name} exclaims. 'I hope your journey was smoother than mine.'",
        lambda: f"{name} pauses for a moment. 'I have much to share, but tell me, what is your business in this village?'",
        lambda: f"The {name} nods slowly. 'Greetings. I am here to help, if you have the coin.'",
    ]
    if category == 'greetings':
        return random.choice(templates)()
    if category == 'trade_related':
        return f"{name} eyes your gear carefully. 'I deal in quality only. Are you buying or just looking?'"
    return f"{name} considers your words deeply. 'Interesting... {player_input} is not something I hear every day.'"

def generate_teacher_response(persona, category, player_input):
    if not call_groq_api or not os.environ.get("GROQ_API_KEY"):
        return generate_heuristic_response(persona, category, player_input)
        
    name = persona.get('id', 'NPC').replace('npc_', '').capitalize()
    desc = persona.get('persona_en', persona.get('description', 'An NPC.'))
    traits = ", ".join(persona.get('traits', ['friendly']))
    
    prompt = f"""You are a creative writer for an RPG game.
Write a single, natural response for the following NPC to the player's input.
The response should be strictly in English and in character. Do not include any out-of-character text, explanations, or quotes surrounding the response. You can include descriptive actions if relevant.

NPC Persona:
Name: {name}
Description: {desc}
Traits: {traits}

Category: {category}

Player says: "{player_input}"

NPC Response:"""

    try:
        response = call_groq_api(prompt, model_id='llama-3.3-70b-versatile', temperature=0.8, max_tokens=150)
        return response.strip(' "\'')
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        return generate_heuristic_response(persona, category, player_input)

if isinstance(personas, dict):
    persona_list = list(personas.values())
elif isinstance(personas, list):
    persona_list = personas
else:
    persona_list = []

if not persona_list:
    persona_list = [{'persona_en': 'You are an NPC.', 'traits': ['friendly'], 'id': 'npc_default'}]

categories = [
    key for key, value in utterances.items()
    if isinstance(value, dict) and isinstance(value.get('en', []), list) and value.get('en')
]
if not categories:
    utterances = {'greetings': {'en': ['Hello!']}}
    categories = ['greetings']


dataset = []
for i in range(500):
    p = random.choice(persona_list)
    c = random.choice(categories)
    q = random.choice(utterances[c].get('en', ['Hello']))
    a = generate_teacher_response(p, c, q)
    ctx = {
        'memories': [],
        'current_emotion': {'description': 'neutral', 'valence': 0.0},
        'knowledge': [],
        'npc_info': {'name': p.get('id', 'NPC'), 'persona': p.get('persona_en', '')},
    }
    prompt = (
        '[INSTRUCTION] Respond strictly in English.\n[CONTEXT]\n'
        + json.dumps(ctx, ensure_ascii=False)
        + '\n\n[PLAYER] '
        + q
        + '\n\n[NPC] '
    )
    dataset.append({'prompt': prompt, 'completion': a})
    
    if (i + 1) % 50 == 0:
        print(f"Generated {i + 1}/500 teacher samples...")

with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(dataset, f, indent=1, ensure_ascii=False)

print(f'Generated {len(dataset)} training samples at {OUTPUT_PATH}')



# ======================================================================
# CODE CELL 7
# ======================================================================
# ============================================================
# Cell 3: Trainer selection and dry-run (+ resume bootstrap)
# ============================================================
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

TRAIN_SCRIPT = 'scripts/train_qlora.py'
if not os.path.exists(TRAIN_SCRIPT):
    raise FileNotFoundError(f'{TRAIN_SCRIPT} not found. Ensure repository files are present.')

IN_KAGGLE = os.path.exists('/kaggle')
WORK_DIR = '/kaggle/working' if os.path.exists('/kaggle/working') else os.getcwd()
OUTPUT_NPC_MODEL = os.path.join(WORK_DIR, 'outputs', 'npc_model') if IN_KAGGLE else 'outputs/npc_model'
os.makedirs(OUTPUT_NPC_MODEL, exist_ok=True)

# Set True to auto-resume from Kaggle dataset checkpoint package.
RESUME_FROM_MODEL150_DATASET = True
MODEL150_DATASET_ROOTS = [
    Path('/kaggle/input/datasets/mphacc3/model150'),
    Path('/kaggle/input/model150'),
]


def _checkpoint_steps(path: Path):
    values = []
    for ckpt in path.glob('checkpoint-*'):
        try:
            values.append(int(ckpt.name.split('-')[-1]))
        except Exception:
            pass
    return sorted(values)


def _find_resume_source() -> Path | None:
    candidates = []
    for root in MODEL150_DATASET_ROOTS:
        if not root.exists():
            continue
        candidates.extend([
            root / 'npc_model',
            root / 'outputs' / 'npc_model',
            root,
        ])
        candidates.extend([p for p in root.glob('**/npc_model') if p.is_dir()])

    for cand in candidates:
        if not cand.exists() or not cand.is_dir():
            continue
        if list(cand.glob('checkpoint-*')):
            return cand
    return None


def stage_resume_checkpoint_if_available() -> None:
    if not IN_KAGGLE or not RESUME_FROM_MODEL150_DATASET:
        return

    src = _find_resume_source()
    if src is None:
        print('No model150 resume dataset found. Training will start from current OUTPUT_NPC_MODEL state.')
        return

    dst = Path(OUTPUT_NPC_MODEL)
    src_steps = _checkpoint_steps(src)
    dst_steps = _checkpoint_steps(dst)

    if dst_steps and src_steps and max(dst_steps) >= max(src_steps):
        print(f'Resume target already has checkpoint-{max(dst_steps)} (>= source checkpoint-{max(src_steps)}).')
        return

    print(f'Staging resume checkpoints from {src} -> {dst}')
    shutil.copytree(src, dst, dirs_exist_ok=True)

    staged_steps = _checkpoint_steps(dst)
    if staged_steps:
        print(f'Resume checkpoint ready: checkpoint-{max(staged_steps)}')
    else:
        print('Warning: no checkpoint-* directories found after staging.')


stage_resume_checkpoint_if_available()

candidates = [
    'data/npc_training_v2.json',
    'data/npc_training_v2.jsonl',
    'data/npc_training.json',
    'data/npc_training.jsonl',
]
TRAIN_DATASET = next((pp for pp in candidates if os.path.exists(pp)), None)
if TRAIN_DATASET is None:
    raise FileNotFoundError('No training dataset found in expected paths.')

print(f'Using train script: {TRAIN_SCRIPT}')
print(f'Using dataset: {TRAIN_DATASET}')
print(f'Output dir: {OUTPUT_NPC_MODEL}')
print(f'Accelerator target: {globals().get("TRAIN_ACCELERATOR", "auto")}')
print(f"Set TRAIN_TOTAL_EPOCHS (current default next cell): {globals().get('TRAIN_TOTAL_EPOCHS', 'auto')}")

# Dry-run to validate config and dataset before long training.
dry_cmd = [
    sys.executable,
    TRAIN_SCRIPT,
    '--data',
    TRAIN_DATASET,
    '--output-dir',
    OUTPUT_NPC_MODEL,
    '--accelerator',
    globals().get('TRAIN_ACCELERATOR', 'auto'),
    '--dry-run',
]
subprocess.check_call(dry_cmd)
print('Dry-run validation completed.')



# ======================================================================
# MARKDOWN CELL 8
# ======================================================================
# ---
# ## 3. 🚀 QLoRA Fine-Tuning



# ======================================================================
# CODE CELL 9
# ======================================================================
# ============================================================
# Cell 4: Execute fine-tuning (TPU/GPU aware) and optional GGUF export
# ============================================================
import glob
import importlib.util
import os
import shutil
import subprocess
import sys

WORK_DIR = '/kaggle/working' if os.path.exists('/kaggle/working') else os.getcwd()
os.makedirs(WORK_DIR, exist_ok=True)

TRAINING_SUCCESS = False
GGUF_EXPORT_SUCCESS = False

train_script = 'scripts/train_qlora.py'
train_data = globals().get('TRAIN_DATASET', 'data/npc_training_v2.json')
accelerator = globals().get('TRAIN_ACCELERATOR', 'auto')
output_dir = globals().get('OUTPUT_NPC_MODEL', 'outputs/npc_model')
if os.path.exists('/kaggle/working') and not os.path.isabs(output_dir):
    output_dir = os.path.join('/kaggle/working', output_dir)
os.makedirs(output_dir, exist_ok=True)

# If resuming from checkpoint-150, keep total epochs > 1 to continue training.
resume_enabled = bool(globals().get('RESUME_FROM_MODEL150_DATASET', False))
TRAIN_TOTAL_EPOCHS = int(globals().get('TRAIN_TOTAL_EPOCHS', 2 if resume_enabled else 1))

if accelerator == 'tpu':
    os.environ.setdefault('PJRT_DEVICE', 'TPU')

train_args = [
    '--data',
    train_data,
    '--output-dir',
    output_dir,
    '--accelerator',
    accelerator,
    '--epochs',
    str(TRAIN_TOTAL_EPOCHS),
    '--max-steps',
    '10',
    '--max-seq-length',
    '256',
    '--learning-rate',
    '2e-4',
    '--gradient-checkpointing',
]

if accelerator == 'cuda':
    train_args += ['--use-4bit', '--batch-size', '2', '--gradient-accumulation-steps', '4']
elif accelerator == 'tpu':
    # TPU cannot use bitsandbytes 4-bit. Increase grad accumulation for stable global batch.
    train_args += ['--no-4bit', '--batch-size', '1', '--gradient-accumulation-steps', '16']
else:
    train_args += ['--no-4bit', '--batch-size', '1', '--gradient-accumulation-steps', '8']

if accelerator == 'tpu' and importlib.util.find_spec('torch_xla.distributed.xla_run') is not None:
    tpu_cores = os.environ.get('NPC_TPU_CORES', '8')
    train_cmd = [
        sys.executable,
        '-m',
        'torch_xla.distributed.xla_run',
        '--num_cores',
        str(tpu_cores),
        train_script,
    ] + train_args
else:
    if accelerator == 'tpu':
        print('torch_xla xla_run launcher not found; using single-process TPU execution.')
    train_cmd = [sys.executable, train_script] + train_args

print('Running training command:')
print(' '.join(train_cmd))
print(f'TRAIN_TOTAL_EPOCHS={TRAIN_TOTAL_EPOCHS}')

try:
    subprocess.check_call(train_cmd)
    TRAINING_SUCCESS = True
    print('Fine-tuning completed.')
except Exception as exc:
    print(f'Warning: fine-tuning failed: {exc}')

if TRAINING_SUCCESS and accelerator != 'tpu' and os.path.exists('scripts/export_gguf.py'):
    # Optional export path. On TPU we skip by default to avoid long CPU merge/convert.
    gguf_out = os.path.join(WORK_DIR, 'npc-phi3.gguf')
    merged_dir = os.path.join(WORK_DIR, 'outputs', 'merged')
    export_cmd = [
        sys.executable,
        'scripts/export_gguf.py',
        '--adapter',
        output_dir,
        '--base-model',
        'microsoft/Phi-3-mini-4k-instruct',
        '--output',
        gguf_out,
        '--merged-dir',
        merged_dir,
    ]
    print('Attempting GGUF export...')
    try:
        subprocess.check_call(export_cmd)
        GGUF_EXPORT_SUCCESS = True
    except Exception as exc:
        print(f'Warning: GGUF export failed: {exc}')

# Ensure any generated GGUF is copied to work dir.
for src in glob.glob('**/*.gguf', recursive=True):
    dst = os.path.join(WORK_DIR, os.path.basename(src))
    if os.path.abspath(src) != os.path.abspath(dst) and not os.path.exists(dst):
        shutil.copy2(src, dst)

if accelerator == 'tpu':
    print('TPU training complete. To export GGUF, run export_gguf.py later on a GPU/CPU runtime.')
print(f'TRAINING_SUCCESS={TRAINING_SUCCESS}, GGUF_EXPORT_SUCCESS={GGUF_EXPORT_SUCCESS}')



# ======================================================================
# MARKDOWN CELL 10
# ======================================================================
# ---
# ## 4. 📦 GGUF Export



# ======================================================================
# CODE CELL 11
# ======================================================================
# ============================================================
# Cell 5: GGUF Export Status
# ============================================================
import glob
import os

WORK_DIR = '/kaggle/working' if os.path.exists('/kaggle/working') else os.getcwd()
OUTPUT_NPC_MODEL = globals().get('OUTPUT_NPC_MODEL', 'outputs/npc_model')

all_ggufs = (
    glob.glob(os.path.join(WORK_DIR, '*.gguf'))
    + glob.glob('/tmp/model_export*.gguf')
    + glob.glob('/tmp/model_export*/**/*.gguf')
    + glob.glob('/tmp/model_export_gguf/*.gguf')
    + glob.glob(os.path.join(OUTPUT_NPC_MODEL, '*.gguf'))
    + glob.glob('*.gguf')
    + glob.glob('**/*.gguf', recursive=True)
)

candidates = [
    f
    for f in all_ggufs
    if os.path.exists(f)
    and os.path.getsize(f) > 200 * 1024 * 1024
    and not any(x in f.lower() for x in ['vocab', 'embedding', 'bge', 'bert'])
]

if candidates:
    trained_model_path = candidates[0]
    print(f'GGUF found: {trained_model_path}')
else:
    trained_model_path = None
    print('Warning: GGUF model not found yet. Ollama registration will be skipped.')



# ======================================================================
# MARKDOWN CELL 12
# ======================================================================
# ---
# ## 5. 🤖 Ollama Serving



# ======================================================================
# CODE CELL 13
# ======================================================================
# ============================================================
# Cell 6: Ollama Serving (Safe Register)
# ============================================================
import glob
import os
import shutil
import subprocess
import time

import requests

OLLAMA_READY = False
ollama_bin = shutil.which('ollama')

if not ollama_bin:
    print('Warning: ollama binary not found. Skipping model registration.')
else:
    print('Starting or checking Ollama server...')
    server_ok = False
    try:
        if requests.get('http://localhost:11434/api/tags', timeout=2).status_code == 200:
            server_ok = True
            print('Ollama is already running.')
    except Exception as exc:
        print(f'Warning: initial Ollama health check failed: {exc}')

    if not server_ok:
        try:
            subprocess.Popen([ollama_bin, 'serve'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            time.sleep(5)
            server_ok = requests.get('http://localhost:11434/api/tags', timeout=5).status_code == 200
        except Exception as exc:
            print(f'Warning: failed to start Ollama: {exc}')

    tm_path = globals().get('trained_model_path')
    if not tm_path or not os.path.exists(tm_path):
        all_ggufs = (
            glob.glob('model_gguf/*.gguf')
            + glob.glob('*.gguf')
            + glob.glob('outputs/*.gguf')
            + glob.glob('/kaggle/working/*.gguf')
        )
        candidates = [
            f
            for f in all_ggufs
            if os.path.exists(f)
            and os.path.getsize(f) > 200 * 1024 * 1024
            and not any(x in f.lower() for x in ['vocab', 'embedding', 'bge', 'bert'])
        ]
        if candidates:
            tm_path = candidates[0]

    if server_ok and tm_path and os.path.exists(tm_path):
        lines = [
            f'FROM {tm_path}',
            'PARAMETER temperature 0.7',
            'PARAMETER stop "[PLAYER]"',
            'PARAMETER stop "[INSTRUCTION]"',
            'PARAMETER stop "[CONTEXT]"',
            'PARAMETER stop "<|end|>"',
            'SYSTEM "You are an NPC. Always respond strictly in English as the [NPC] speaker. Do not repeat the prompt."',
        ]
        with open('Modelfile', 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        print(f'Registering model npc-ai from: {tm_path}')
        result = subprocess.run(
            [ollama_bin, 'create', 'npc-ai', '-f', 'Modelfile'],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            OLLAMA_READY = True
            print('Ollama model registration succeeded.')
        else:
            print('Warning: Ollama model registration failed:')
            print(result.stderr[-500:])
    else:
        print('Warning: Ollama server/model prerequisites not satisfied; skipping registration.')




# ======================================================================
# MARKDOWN CELL 14
# ======================================================================
# ---
# ## 6. 🎮 Integrated Demo (Enhanced)



# ======================================================================
# CODE CELL 15
# ======================================================================
# ============================================================
# Cell 7: Integrated Demo (Clean Turns)
# ============================================================
import json

import requests


def query_npc(player_input):
    if not globals().get('OLLAMA_READY', False):
        return '[Ollama model is not ready]'

    ctx = {
        'memories': [],
        'current_emotion': {'description': 'neutral', 'valence': 0.0},
        'npc_info': {'name': 'Blacksmith', 'persona': 'A friendly blacksmith.'},
    }
    prompt = (
        '[INSTRUCTION] Respond strictly in English.\n[CONTEXT]\n'
        + json.dumps(ctx)
        + '\n\n[PLAYER] '
        + player_input
        + '\n\n[NPC] '
    )

    try:
        payload = {
            'model': 'npc-ai',
            'prompt': prompt,
            'stream': False,
            'options': {'stop': ['[PLAYER]', '[INSTRUCTION]', '<|end|>']},
        }
        res = requests.post('http://localhost:11434/api/generate', json=payload, timeout=60)
        if res.status_code == 200:
            text = res.json().get('response', '[No response]')
            return text.split('[NPC]')[-1].strip()
        return f'[Error {res.status_code}]'
    except Exception as exc:
        return f'[Error: {exc}]'


for inp in ['Hello! I am new here.', 'What is the curse?']:
    print(f'Player: {inp}\nNPC: {query_npc(inp)}\n')



# ======================================================================
# MARKDOWN CELL 16
# ======================================================================
# ---
# ## 7. 📊 Quality Evaluation



# ======================================================================
# CODE CELL 17
# ======================================================================
print('Evaluating responses...')
# Simplified evaluation loop
if 'query_npc' not in globals():
    print('query_npc is unavailable; skipping evaluation.')
else:
    test_queries = ['Hello!', 'Who are you?', 'Tell me a story.']
    for q in test_queries:
        resp = query_npc(q)
        print(f'Q: {q}\nA: {resp[:80]}...\n')



# ======================================================================
# MARKDOWN CELL 18
# ======================================================================
# ---
# ## 8. 🛠️ C++ Engine Compilation



# ======================================================================
# CODE CELL 19
# ======================================================================
# ============================================================
# Cell 10: C++ Engine Compilation (Optimized)
# ============================================================
import os
import subprocess

if os.path.exists('cpp'):
    os.makedirs('cpp/build', exist_ok=True)
    jobs = max(1, os.cpu_count() or 1)

    def run_build(extra_args=None):
        extra_args = extra_args or []
        cfg_cmd = ['cmake', '..'] + list(extra_args)
        subprocess.check_call(cfg_cmd, cwd='cpp/build')
        subprocess.check_call(['cmake', '--build', '.', f'-j{jobs}'], cwd='cpp/build')

    try:
        run_build()
        print('Compilation successful.')
    except Exception as exc:
        print(f'Warning: default C++ compilation failed: {exc}')
        print('Retrying with USEARCH disabled for Kaggle compatibility...')
        try:
            run_build(['-DNPC_USE_USEARCH=OFF'])
            print('Compilation successful with NPC_USE_USEARCH=OFF.')
        except Exception as exc2:
            print(f'Warning: C++ compilation failed after retry: {exc2}')
else:
    print('Warning: cpp/ not found.')



# ======================================================================
# MARKDOWN CELL 20
# ======================================================================
# ---
# ## 9. 📈 Performance Benchmarking



# ======================================================================
# CODE CELL 21
# ======================================================================
# ============================================================
# Cell 11: C++ Engine Benchmarks
# ============================================================
import os
import subprocess
import sys

# Extra args passed per-benchmark. ablation_suite uses mock mode so it
# does not require real ONNX model files (tokenizer.model, etc.).
BENCHMARK_EXTRA_ARGS = {
    'ablation_suite': ['--model-dir', 'models/phi3_onnx_official/cpu_and_mobile/cpu-int4-rtn-block-32', '--runs', '2'],
}


def _find_benchmark(bench):
    """Return first found path for a C++ benchmark binary."""
    candidates = [
        f'cpp/build/Release/{bench}.exe',  # Windows MSVC
        f'cpp/build/{bench}.exe',          # Windows fallback
        f'cpp/build/Release/{bench}',      # Linux release
        f'cpp/build/{bench}',              # Linux debug / other
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


if os.path.exists('cpp/build'):
    print('Running C++ engine benchmarks...')
    benchmarks = ['bench_engine', 'bench_memory', 'bench_retrieval', 'ablation_suite']

    for bench in benchmarks:
        path = _find_benchmark(bench)
        if path:
            print(f'\nExecuting {bench} ({path})...')
            print('-' * 40)
            try:
                extra = BENCHMARK_EXTRA_ARGS.get(bench, [])
                res = subprocess.run(
                    [path] + extra,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                print(res.stdout)
                if res.returncode != 0 and res.stderr:
                    print(f'Stderr (rc={res.returncode}): {res.stderr[-1000:]}')
            except subprocess.TimeoutExpired:
                print(f'Warning: {bench} timed out after 5 minutes.')
            except Exception as exc:
                print(f'Warning: failed to run {bench}: {exc}')
        else:
            print(f'Warning: benchmark binary not found for {bench} (checked Release/*.exe and build/*).')
else:
    print('Warning: cpp/build not found. Run the compilation cell first.')



# ======================================================================
# MARKDOWN CELL 22
# ======================================================================
# ---
# ## 10. 📊 Ablation Study Visualization



# ======================================================================
# CODE CELL 23
# ======================================================================
# ============================================================
# Cell 12: Visualize Ablation Results
# ============================================================
import os, json

try:
    import pandas as pd
    import matplotlib.pyplot as plt
except Exception as exc:
    pd = None
    plt = None
    print(f'Warning: plotting libraries unavailable: {exc}')

results_path = 'cpp/build/ablation_results.json'
if pd is None or plt is None:
    print('Skipping ablation visualization because pandas/matplotlib are missing.')
elif os.path.exists(results_path):
    print(f"📈 Loading ablation results from {results_path}...")
    with open(results_path, 'r') as f:
        data = json.load(f)

    records = []
    if isinstance(data, list):
        for item in data:
            config = item.get("config_name", "Unknown")
            metrics = item.get("results", {})
            records.append({
                'Configuration': config,
                'Latency p95 (ms)': metrics.get('latency_p95_ms', 0),
                'Throughput (tok/s)': metrics.get('throughput_tokens_per_sec', 0),
                'Memory (MB)': metrics.get('memory_usage_mb', 0)
            })
    else:
        for config, metrics in data.items():
            records.append({
                'Configuration': config,
                'Latency p95 (ms)': metrics.get('latency_p95_ms', 0),
                'Throughput (tok/s)': metrics.get('throughput_tok_s', 0),
                'Memory (MB)': metrics.get('peak_memory_mb', 0)
            })

    df = pd.DataFrame(records)
    # display(df) # Commented out for standalone script robustness
    print(df)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    df.plot(x='Configuration', y='Latency p95 (ms)', kind='bar', ax=axes[0], color='salmon', legend=False)
    axes[0].set_title('95th Percentile Latency (Lower is Better)')
    axes[0].set_ylabel('Milliseconds (ms)')
    axes[0].tick_params(axis='x', rotation=45)

    df.plot(x='Configuration', y='Throughput (tok/s)', kind='bar', ax=axes[1], color='skyblue', legend=False)
    axes[1].set_title('Generation Throughput (Higher is Better)')
    axes[1].set_ylabel('Tokens per Second')
    axes[1].tick_params(axis='x', rotation=45)

    df.plot(x='Configuration', y='Memory (MB)', kind='bar', ax=axes[2], color='lightgreen', legend=False)
    axes[2].set_title('Peak Memory Usage (Lower is Better)')
    axes[2].set_ylabel('Megabytes (MB)')
    axes[2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()
else:
    print(f"⚠️ Ablation results not found at {results_path}. Make sure Cell 11 ran successfully.")




# ======================================================================
# MARKDOWN CELL 24
# ======================================================================
# ## Proposal Evaluation (Batched)
# Generate expanded scenarios and run proposal evaluation in batches for Kaggle stability.



# ======================================================================
# CODE CELL 25
# ======================================================================
import os
import subprocess
import sys

hf_cache = "/kaggle/working/hf_cache" if os.path.exists("/kaggle/working") else os.path.abspath("hf_cache")
os.makedirs(hf_cache, exist_ok=True)

subprocess.check_call(
    [
        sys.executable,
        "scripts/generate_proposal_scenarios_large.py",
        "--variants-per-base",
        "14",
        "--output",
        "data/proposal_eval_scenarios_large.jsonl",
    ]
)

cmd = [
    sys.executable,
    "scripts/run_proposal_alignment_eval_batched.py",
    "--scenarios",
    "data/proposal_eval_scenarios_large.jsonl",
    "--batch-size",
    "28",
    "--repeats",
    "1",
    "--max-tokens",
    "80",
    "--temperature",
    "0.2",
    "--baseline-models",
    "phi3:latest",
    "--bertscore-model-type",
    "roberta-large",
    "--bertscore-batch-size",
    "16",
    "--bertscore-cache-dir",
    hf_cache,
]
print("Running:", " ".join(cmd))
subprocess.check_call(cmd)



# ======================================================================
# MARKDOWN CELL 26
# ======================================================================
# ## Human Evaluation Pack (Optional)
# Build blind multi-rater annotation files from the latest proposal run.



# ======================================================================
# CODE CELL 27
# ======================================================================
import pathlib
import subprocess
import sys

proposal_root = pathlib.Path("artifacts/proposal")
run_dirs = sorted([p for p in proposal_root.iterdir() if p.is_dir()]) if proposal_root.exists() else []
if not run_dirs:
    raise RuntimeError("No proposal runs found under artifacts/proposal. Run proposal eval first.")
latest_run = run_dirs[-1]

subprocess.check_call(
    [
        sys.executable,
        "scripts/build_human_eval_pack.py",
        "--run-dir",
        str(latest_run),
        "--annotators",
        "annotator_1,annotator_2,annotator_3",
        "--shared-ratio",
        "0.35",
    ]
)



# ======================================================================
# MARKDOWN CELL 28
# ======================================================================
# ## Publication Benchmark Suite
# Run non-mock benchmark suite with retrieval security checks.



# ======================================================================
# CODE CELL 29
# ======================================================================
import subprocess
import sys

cmd = [
    sys.executable,
    "scripts/run_publication_benchmark_suite.py",
    "--repeats",
    "1",
    "--max-tokens",
    "64",
    "--temperature",
    "0.2",
    "--run-security-benchmark",
    "--run-security-spoofed-benchmark",
]
print("Running:", " ".join(cmd))
subprocess.check_call(cmd)



# ======================================================================
# MARKDOWN CELL 30
# ======================================================================
# ## Proposal Quality Gate
# Evaluate whether latest proposal/publication artifacts satisfy the quality bar.



# ======================================================================
# CODE CELL 31
# ======================================================================
import subprocess
import sys

cmd = [
    sys.executable,
    "scripts/proposal_quality_gate.py",
    "--proposal-run",
    "latest",
    "--publication-run",
    "latest",
    "--require-security-benchmark",
]
print("Running:", " ".join(cmd))
subprocess.check_call(cmd)



# ======================================================================
# MARKDOWN CELL 32
# ======================================================================
# ## Proposal Run Comparison (Latest vs Previous)
# Check whether the latest run actually improved key metrics and whether CI overlap suggests a stable gain.



# ======================================================================
# CODE CELL 33
# ======================================================================
import json
from pathlib import Path

PROPOSAL_ROOT = Path("artifacts/proposal")
ARM = "proposed_contextual_controlled"
METRICS = ["overall_quality", "context_relevance", "persona_consistency", "bertscore_f1"]


def ci_overlap(a_low, a_high, b_low, b_high):
    return max(float(a_low), float(b_low)) <= min(float(a_high), float(b_high))


run_dirs = sorted(
    [p for p in PROPOSAL_ROOT.iterdir() if p.is_dir() and (p / "summary.json").exists()],
    key=lambda p: p.name,
) if PROPOSAL_ROOT.exists() else []

if len(run_dirs) < 2:
    raise RuntimeError("Need at least two proposal runs with summary.json under artifacts/proposal.")

prev_run, new_run = run_dirs[-2], run_dirs[-1]
prev = json.loads((prev_run / "summary.json").read_text(encoding="utf-8"))
new = json.loads((new_run / "summary.json").read_text(encoding="utf-8"))

if ARM not in prev or ARM not in new:
    raise RuntimeError(f"Arm '{ARM}' not present in both summaries.")

print(f"Previous run: {prev_run.name}")
print(f"Latest run  : {new_run.name}")
print(f"Arm         : {ARM}")
print()
print("metric	prev_mean	new_mean	delta	ci_overlap")

for metric in METRICS:
    p = prev[ARM].get(metric)
    n = new[ARM].get(metric)
    if not p or not n:
        print(f"{metric}	N/A	N/A	N/A	N/A")
        continue

    p_mean = float(p.get("mean", float('nan')))
    n_mean = float(n.get("mean", float('nan')))
    delta = n_mean - p_mean

    p_low = float(p.get("ci95_low", p_mean))
    p_high = float(p.get("ci95_high", p_mean))
    n_low = float(n.get("ci95_low", n_mean))
    n_high = float(n.get("ci95_high", n_mean))
    overlap = ci_overlap(p_low, p_high, n_low, n_high)

    print(f"{metric}	{p_mean:.4f}	{n_mean:.4f}	{delta:+.4f}	{overlap}")



# ======================================================================
# MARKDOWN CELL 34
# ======================================================================
# ## Full Artifact Checkout (Recommended)
# Run the complete proposal/publication pipeline and emit a single manifest with all output paths.



# ======================================================================
# CODE CELL 35
# ======================================================================
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Option: set True to skip keyword/random ablation baselines in publication retrieval metrics.
SKIP_ABLATION_BASELINES = False
# Option: continue as dry-run if Ollama host/models are unavailable.
ALLOW_MISSING_OLLAMA = True


def has_repo_scripts(root: Path) -> bool:
    return (root / "scripts" / "run_kaggle_full_results.py").exists()


def ensure_repo_root() -> Path:
    cwd = Path.cwd()
    if has_repo_scripts(cwd):
        return cwd

    if Path("/kaggle").exists():
        dataset_candidates = [
            Path("/kaggle/input/datasets/mphacc3/npc-ai"),
            Path("/kaggle/input/npc-ai"),
        ]
        for base in dataset_candidates:
            for candidate in (base, base / "npc-ai", base / "NPC AI", base / "NPC_AI"):
                if not has_repo_scripts(candidate):
                    continue
                target = Path("/kaggle/working/npc-ai")
                if not has_repo_scripts(target):
                    shutil.copytree(candidate, target, dirs_exist_ok=True)
                os.chdir(target)
                print(f"Running from repo root: {target}")
                return target

    raise FileNotFoundError(
        "Could not locate scripts/run_kaggle_full_results.py. "
        "On Kaggle, attach dataset mphacc3/npc-ai."
    )


ensure_repo_root()

cmd = [
    sys.executable,
    "scripts/run_kaggle_full_results.py",
    "--host",
    "http://127.0.0.1:11434",
]
if SKIP_ABLATION_BASELINES:
    cmd.append("--skip-ablation-baselines")
if ALLOW_MISSING_OLLAMA:
    cmd.append("--allow-missing-ollama")

print("Running:", " ".join(cmd))
subprocess.check_call(cmd)


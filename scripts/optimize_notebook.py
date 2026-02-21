import json
import os

notebook_path = r'f:\NPC AI\notebooks\NPC_AI_Complete_Pipeline.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_cells = []

for cell in nb['cells']:
    # Remove obvious duplicates by ID or source snippet
    source_str = "".join(cell.get('source', []))
    
    # Skip redundant fine-tuning execution cells
    if 'scripts/train_unsloth.py' in source_str and 'avoid memory fragmentation' not in source_str:
        continue
    
    # Optimize GGUF Export cell (add cleanup)
    if 'model.save_pretrained_gguf' in source_str:
        cell['source'] = [
            "# ============================================================\n",
            "# Cell 4: GGUF Export (with Memory Management)\n",
            "# ============================================================\n",
            "from unsloth import FastLanguageModel\n",
            "import os, glob, gc, torch\n",
            "\n",
            "# Clear VRAM before loading for export\n",
            "gc.collect()\n",
            "torch.cuda.empty_cache()\n",
            "\n",
            "model_name = \"outputs/npc_model\"\n",
            "save_path = \"model_gguf\"\n",
            "if os.path.exists(model_name):\n",
            "    model, tokenizer = FastLanguageModel.from_pretrained(model_name, load_in_4bit=True)\n",
            "    print('\ud83d\udce6 Exporting to F16 GGUF...')\n",
            "    model.save_pretrained_gguf(save_path, tokenizer, quantization_method = \"f16\")\n",
            "    \n",
            "    # Cleanup immediately after export\n",
            "    del model\n",
            "    del tokenizer\n",
            "    gc.collect()\n",
            "    torch.cuda.empty_cache()\n",
            "    \n",
            "    # Robustly find the exported GGUF file\n",
            "    gguf_files = glob.glob(f\"{save_path}*/**/*.gguf\", recursive=True) + glob.glob(f\"{save_path}*/*.gguf\")\n",
            "    if gguf_files:\n",
            "        trained_model_path = gguf_files[0]\n",
            "        print(f'\u2705 GGUF exported and VRAM cleared: {trained_model_path}')\n",
            "    else:\n",
            "        trained_model_path = os.path.join(save_path, \"phi-3-mini-4k-instruct.F16.gguf\")\n",
            "        print(f'\u26a0\ufe0f GGUF not found via glob. Fallback: {trained_model_path}')\n",
            "else:\n",
            "    print('\u26a0\ufe0f Trained model not found. Using pre-trained for demo.')\n",
            "    trained_model_path = \"unsloth/Phi-3-mini-4k-instruct-gguf\"\n"
        ]

    # Optimize Fine-tuning cell
    if 'scripts/train_unsloth.py' in source_str and 'avoid memory fragmentation' in source_str:
        cell['source'] = [
            "# ============================================================\n",
            "# Cell 4: Execute Fine-tuning (VRAM Safe)\n",
            "# ============================================================\n",
            "import subprocess, sys, os, torch, gc\n",
            "print('\ud83d\ude80 Starting fine-tuning...')\n",
            "# Call standalone script to keep main notebook process clean\n",
            "subprocess.check_call([sys.executable, 'scripts/train_unsloth.py', \n",
            "                    '--dataset', 'data/npc_training_v2.json',\n",
            "                    '--output_dir', 'outputs/npc_model'])\n",
            "\n",
            "# VRAM cleanup after subprocess exits\n",
            "gc.collect()\n",
            "torch.cuda.empty_cache()\n",
            "print('\u2705 Fine-tuning complete and VRAM cleared.')"
        ]

    new_cells.append(cell)

nb['cells'] = new_cells

# Write with ensure_ascii=True to avoid surrogate issues in JSON
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=True)

print("Notebook optimized successfully.")

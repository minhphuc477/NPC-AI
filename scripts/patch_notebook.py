"""Patch NPC_Project_Complete_Kaggle_FINAL.ipynb with fixes:
1. Add Ollama installation before 'ollama serve'
2. Fix GGUF export path (Unsloth appends '_gguf' to dir name)
3. Fix GGUF glob pattern with fallbacks

Run: python scripts/patch_notebook.py
"""
import json
import os

NOTEBOOK_PATH = os.path.join(os.path.dirname(__file__), "..", "notebooks", "NPC_Project_Complete_Kaggle_FINAL.ipynb")

def patch():
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = cell["source"]
        joined = "".join(source)

        # Fix 1: GGUF export path
        if 'model.save_pretrained_gguf("model_gguf"' in joined:
            new_source = []
            skip_next_raise = False
            for line in source:
                if 'model.save_pretrained_gguf("model_gguf"' in line:
                    new_source.append('    # Note: Unsloth appends "_gguf" to dir name, so "model" -> "model_gguf"\n')
                    new_source.append('    model.save_pretrained_gguf("model", tokenizer, quantization_method = "f16")\n')
                elif 'gguf_files = glob.glob("model_gguf/*.gguf")' in line:
                    new_source.append(line)
                    new_source.append('    if not gguf_files:\n')
                    new_source.append('        gguf_files = glob.glob("model/*.gguf") + glob.glob("model_gguf_gguf/*.gguf")\n')
                    new_source.append('    if not gguf_files:\n')
                    new_source.append('        gguf_files = glob.glob("**/*.gguf", recursive=True)\n')
                elif "raise FileNotFoundError" in line and "No GGUF" in line:
                    new_source.append(line)
                elif 'print("Training Complete' in line or "model-unsloth.f16.gguf" in line:
                    new_source.append('    print(f"Training Complete & Model Saved to {trained_model_path}")\n')
                else:
                    new_source.append(line)
            cell["source"] = new_source
            print("  [OK] Fixed GGUF export path")

        # Fix 2: Add Ollama installation
        if 'ollama_process = subprocess.Popen(["ollama", "serve"]' in joined:
            new_source = []
            seen_imports = set()
            skip_dup_comment = False
            for line in source:
                stripped = line.strip()
                # Deduplicate imports
                if stripped in ("import subprocess", "import time"):
                    if stripped in seen_imports:
                        continue
                    seen_imports.add(stripped)
                # Skip duplicate "# 1. Start Server" comment
                if "# 1. Start Server in background" in line:
                    if skip_dup_comment:
                        continue
                    skip_dup_comment = True
                # Insert install before the Popen line
                if 'ollama_process = subprocess.Popen(["ollama", "serve"]' in line:
                    new_source.append("\n")
                    new_source.append("# 0. Install Ollama (not pre-installed on Kaggle)\n")
                    new_source.append('print("Installing Ollama...")\n')
                    new_source.append("!curl -fsSL https://ollama.com/install.sh | sh\n")
                    new_source.append('print("Ollama installed!")\n')
                    new_source.append("\n")
                new_source.append(line)
            cell["source"] = new_source
            print("  [OK] Added Ollama installation step")

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=4, ensure_ascii=False)

    print(f"\nPatched: {NOTEBOOK_PATH}")

if __name__ == "__main__":
    patch()

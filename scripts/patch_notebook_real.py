#!/usr/bin/env python3
import json
import os

NOTEBOOK_PATH = "notebooks/NPC_AI_Complete_Pipeline.ipynb"

def patch_notebook():
    if not os.path.exists(NOTEBOOK_PATH):
        print(f"Error: {NOTEBOOK_PATH} not found.")
        return

    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    patched = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            source = cell.get("source", [])
            # Look for the benchmark cell
            if any("Cell 11:" in line or "C++ Engine Benchmarks" in line for line in source):
                new_source = []
                for line in source:
                    # Replace --mock-mode 1 with the real model path
                    if '["--mock-mode", "1"]' in line:
                        line = line.replace('["--mock-mode", "1"]', '["--model-dir", "models/phi3_onnx_official/cpu_and_mobile/cpu-int4-rtn-block-32"]')
                        patched = True
                    new_source.append(line)
                cell["source"] = new_source

    if patched:
        with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print("Successfully patched notebook to use real model.")
    else:
        print("Could not find the target string to replace in the notebook.")

if __name__ == "__main__":
    patch_notebook()

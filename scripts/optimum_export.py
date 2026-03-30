"""
optimum_export.py - Export merged model to ONNX using Optimum Python API
"""

import importlib
import torch
from pathlib import Path

def run_export(model_path, output_path):
    optimum_mod = importlib.import_module("optimum.exporters.onnx")
    main_export = getattr(optimum_mod, "main_export")
    print(f"Starting Optimum export for: {model_path}")
    print(f"Target directory: {output_path}")
    
    # Run the export
    # We use task="causal-lm" for Phi-3
    main_export(
        model_name_or_path=model_path,
        output=output_path,
        task="causal-lm",
        trust_remote_code=True,
        # device="cpu", # Defaults to CPU
    )
    
    print("✓ Export finished.")

if __name__ == "__main__":
    model_dir = "merged_model_extreme"
    output_dir = "onnx_models"
    
    # Ensure output dir exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    run_export(model_dir, output_dir)

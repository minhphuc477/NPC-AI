"""
save_merged_model.py - Merge LoRA adapter into base model and save locally
"""

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sys
from pathlib import Path

# Add core to path
sys.path.append(str(Path(__file__).parent.parent))
from core.config import BASE_MODEL, ADAPTER_PATH

def save_merged(base_model_name, adapter_path, output_dir):
    print(f"Loading base model: {base_model_name}")
    
    # Load base model
    print("Loading base model with disk offload...")
    import os
    os.makedirs("offload", exist_ok=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        offload_folder="offload",
        trust_remote_code=True
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    
    # Load adapter
    if adapter_path and Path(adapter_path).exists():
        print(f"Loading adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        
        print("Merging adapter weights...")
        model = model.merge_and_unload()
        
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print("No adapter found, saving base model only.")
        
    print(f"Saving merged model to: {output_dir}")
    # Sharding helps with memory peaks during saving
    model.save_pretrained(output_dir, max_shard_size="500MB")
    tokenizer.save_pretrained(output_dir)
    
    # Ensure remote code is copied
    from huggingface_hub import hf_hub_download
    import os
    import shutil
    
    print("Copying remote code files...")
    remote_code_files = ["modeling_phi3.py", "configuration_phi3.py"]
    for filename in remote_code_files:
        try:
            cached_path = hf_hub_download(repo_id=base_model_name, filename=filename)
            shutil.copy(cached_path, os.path.join(output_dir, filename))
            print(f"✓ Copied {filename}")
        except Exception as e:
            print(f"! Could not copy {filename}: {e}")

    print("✓ Model saved successfully.")

if __name__ == "__main__":
    output_directory = "merged_model"
    save_merged(BASE_MODEL, ADAPTER_PATH, output_directory)

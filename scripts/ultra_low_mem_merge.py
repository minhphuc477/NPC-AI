"""
ultra_low_mem_merge.py - Tensor-by-tensor merging for low-RAM environments
"""

import torch
import os
import json
import shutil
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file, load_file
from huggingface_hub import hf_hub_download

def ultra_low_mem_merge(base_model_id, adapter_path, output_dir):
    print(f"Starting ultra-low-mem merge: {base_model_id} + {adapter_path}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load adapter weights & config (small, fits in RAM)
    print("Loading adapter...")
    with open(os.path.join(adapter_path, "adapter_config.json"), "r") as f:
        config = json.load(f)
    
    lora_r = config["r"]
    lora_alpha = config["lora_alpha"]
    scaling = lora_alpha / lora_r
    
    adapter_weights = load_file(os.path.join(adapter_path, "adapter_model.safetensors"))
    
    # 2. Get base model shard index
    print("Fetching base model shard information...")
    index_file = hf_hub_download(repo_id=base_model_id, filename="model.safetensors.index.json")
    with open(index_file, "r") as f:
        shard_data = json.load(f)
    
    weight_map = shard_data["weight_map"]
    shards = sorted(list(set(weight_map.values())))
    
    # 3. Process shards one by one, and tensors one by one
    for i, shard_name in enumerate(shards):
        print(f"Processing shard {i+1}/{len(shards)}: {shard_name}...")
        shard_path = hf_hub_download(repo_id=base_model_id, filename=shard_name)
        
        updated_shard_weights = {}
        
        # Open shard lazily
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            tensors_in_shard = f.keys()
            for name in tensors_in_shard:
                param = f.get_tensor(name)
                
                # Check for LoRA
                lora_base_name = f"base_model.model.{name.replace('.weight', '')}"
                lora_a_name = f"{lora_base_name}.lora_A.weight"
                lora_b_name = f"{lora_base_name}.lora_B.weight"
                
                if lora_a_name in adapter_weights and lora_b_name in adapter_weights:
                    print(f"  Merging LoRA into {name}...")
                    lora_a = adapter_weights[lora_a_name].to(torch.float32)
                    lora_b = adapter_weights[lora_b_name].to(torch.float32)
                    
                    # W_new = W_old + (B @ A) * scaling
                    delta = (lora_b @ lora_a) * scaling
                    param_fp32 = param.to(torch.float32)
                    param_fp32 += delta
                    updated_shard_weights[name] = param_fp32.to(torch.float16)
                    
                    # Clean up temp large tensors
                    del delta
                    del param_fp32
                else:
                    updated_shard_weights[name] = param.to(torch.float16)
        
        # Save updated shard
        save_file(updated_shard_weights, os.path.join(output_dir, shard_name))
        
        # Intensive cleanup
        del updated_shard_weights
        import gc
        gc.collect()

    # 4. Copy required files
    print("Copying configuration and code files...")
    hf_files = ["config.json", "generation_config.json", "modeling_phi3.py", "configuration_phi3.py", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.model"]
    
    for f in hf_files:
        try:
            # Try adapter dir first (for updated tokenizers)
            src = os.path.join(adapter_path, f)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(output_dir, f))
                print(f"✓ Copied {f} from adapter")
            else:
                path = hf_hub_download(repo_id=base_model_id, filename=f)
                shutil.copy(path, os.path.join(output_dir, f))
                print(f"✓ Copied {f} from HF Hub")
        except Exception as e:
            pass
            
    # Save index
    with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(shard_data, f, indent=2)

    print("\n✓ Ultra-low-mem merge complete!")

if __name__ == "__main__":
    ultra_low_mem_merge("microsoft/Phi-3-mini-4k-instruct", "adapter_multiturn", "merged_model_final")

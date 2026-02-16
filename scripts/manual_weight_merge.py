"""
manual_weight_merge.py - Memory-efficient weight merging for Phi-3 + LoRA
"""

import torch
import os
import json
import shutil
from pathlib import Path
from safetensors.torch import load_file, save_file
from huggingface_hub import hf_hub_download, list_repo_files

def manual_merge(base_model_id, adapter_path, output_dir):
    print(f"Starting manual merge: {base_model_id} + {adapter_path}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load adapter weights & config
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
    
    # 3. Process shards one by one
    new_weight_map = {}
    
    for i, shard_name in enumerate(shards):
        print(f"Processing shard {i+1}/{len(shards)}: {shard_name}...")
        shard_path = hf_hub_download(repo_id=base_model_id, filename=shard_name)
        weights = load_file(shard_path)
        
        updated_weights = {}
        for name, param in weights.items():
            # Check if this parameter has LoRA weights
            # LoRA names look like: base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
            # Base names look like: model.layers.0.self_attn.q_proj.weight
            
            # Map base name to potential LoRA names
            # LoRA weights target specific modules (e.g., q_proj)
            # We need to find the matching A and B matrices
            
            merged_param = param.to(torch.float32) # Work in float32 for precision
            
            # Search for matching LoRA weights
            # Example: name = "model.layers.0.self_attn.q_proj.weight"
            # Target = "base_model.model." + name.replace(".weight", "") + ".lora_A.weight"
            try:
                # This logic assumes standard PEFT naming
                lora_base_name = f"base_model.model.{name.replace('.weight', '')}"
                lora_a_name = f"{lora_base_name}.lora_A.weight"
                lora_b_name = f"{lora_base_name}.lora_B.weight"
                
                if lora_a_name in adapter_weights and lora_b_name in adapter_weights:
                    print(f"  Adjusting {name}...")
                    lora_a = adapter_weights[lora_a_name].to(torch.float32)
                    lora_b = adapter_weights[lora_b_name].to(torch.float32)
                    
                    # LoRA: W_new = W_old + (B @ A) * scaling
                    # Note: B @ A for linear layers
                    delta = (lora_b @ lora_a) * scaling
                    merged_param += delta
            except Exception as e:
                print(f"  Warning: error processing LoRA for {name}: {e}")
            
            updated_weights[name] = merged_param.to(torch.float16)
            new_weight_map[name] = shard_name
        
        # Save updated shard
        save_file(updated_weights, os.path.join(output_dir, shard_name))
        del weights
        del updated_weights
        import gc
        gc.collect()

    # 4. Copy config and other files
    print("Copying configuration files...")
    hf_files = list_repo_files(base_model_id)
    configs = [f for f in hf_files if f.endswith(".json") and "model" not in f]
    configs += ["modeling_phi3.py", "configuration_phi3.py", "tokenizer.model"]
    
    for f in configs:
        try:
            path = hf_hub_download(repo_id=base_model_id, filename=f)
            shutil.copy(path, os.path.join(output_dir, f))
            print(f"✓ Copied {f}")
        except Exception as e:
            pass
            
    # Copy adapter/tokenizer files if they exist locally and are more recent
    for f in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        src = os.path.join(adapter_path, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(output_dir, f))
            print(f"✓ Updated {f} from adapter dir")

    # Update index file
    with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(shard_data, f, indent=2)

    print("\n✓ Manual merge complete!")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    manual_merge("microsoft/Phi-3-mini-4k-instruct", "adapter_multiturn", "merged_model_manual")

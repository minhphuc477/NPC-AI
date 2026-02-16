"""
extreme_sharded_merge.py - Merge tensors and save in many small shards
"""

import torch
import os
import json
import shutil
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file, load_file
from huggingface_hub import hf_hub_download

def extreme_sharded_merge(base_model_id, adapter_path, output_dir):
    print(f"Starting extreme sharded merge: {base_model_id} + {adapter_path}")
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
    
    # 3. Process tensors and save in many small shards
    new_weight_map = {}
    current_shard_weights = {}
    current_shard_idx = 1
    tensors_per_shard = 5 # Very small to be safe
    
    for i, shard_name in enumerate(shards):
        print(f"Reading base shard {i+1}/{len(shards)}: {shard_name}...")
        shard_path = hf_hub_download(repo_id=base_model_id, filename=shard_name)
        
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for name in f.keys():
                param = f.get_tensor(name)
                
                # Merge LoRA
                lora_base_name = f"base_model.model.{name.replace('.weight', '')}"
                lora_a_name = f"{lora_base_name}.lora_A.weight"
                lora_b_name = f"{lora_base_name}.lora_B.weight"
                
                if lora_a_name in adapter_weights and lora_b_name in adapter_weights:
                    print(f"  Merging {name}...")
                    lora_a = adapter_weights[lora_a_name].to(torch.float32)
                    lora_b = adapter_weights[lora_b_name].to(torch.float32)
                    delta = (lora_b @ lora_a) * scaling
                    param = (param.to(torch.float32) + delta).to(torch.float16)
                    del delta
                else:
                    param = param.to(torch.float16)
                
                # Add to current micro-shard
                current_shard_weights[name] = param
                
                # If shard is full, save it
                if len(current_shard_weights) >= tensors_per_shard:
                    new_shard_name = f"model-{current_shard_idx:05d}-of-micro.safetensors"
                    save_file(current_shard_weights, os.path.join(output_dir, new_shard_name))
                    for k in current_shard_weights.keys():
                        new_weight_map[k] = new_shard_name
                    
                    current_shard_weights = {}
                    current_shard_idx += 1
                    import gc
                    gc.collect()

    # Save remaining weights
    if current_shard_weights:
        new_shard_name = f"model-{current_shard_idx:05d}-of-micro.safetensors"
        save_file(current_shard_weights, os.path.join(output_dir, new_shard_name))
        for k in current_shard_weights.keys():
            new_weight_map[k] = new_shard_name

    # 4. Save metadata
    print("Saving metadata...")
    # Copy essential files
    hf_files = ["config.json", "generation_config.json", "modeling_phi3.py", "configuration_phi3.py", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.model"]
    for f in hf_files:
        try:
            # Try adapter dir first
            src = os.path.join(adapter_path, f)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(output_dir, f))
            else:
                path = hf_hub_download(repo_id=base_model_id, filename=f)
                shutil.copy(path, os.path.join(output_dir, f))
        except Exception as e:
        logger.warning(f"Error: {e}")
        pass

    # Save custom index
    with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": shard_data["metadata"], "weight_map": new_weight_map}, f, indent=2)

    print(f"\n✓ Extreme sharded merge complete! ({current_shard_idx} shards created)")

if __name__ == "__main__":
    extreme_sharded_merge("microsoft/Phi-3-mini-4k-instruct", "adapter_multiturn", "merged_model_extreme")

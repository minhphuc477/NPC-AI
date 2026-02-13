#!/usr/bin/env python3
"""
INT4 Quantization Script
Quantizes Phi-3 model to INT4 using GPTQ/AWQ for faster inference
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from pathlib import Path

def quantize_to_int4_gptq(
    model_name: str = "microsoft/Phi-3-mini-4k-instruct",
    output_dir: str = "models/phi3-int4-gptq",
    bits: int = 4,
    group_size: int = 128
):
    """
    Quantize model to INT4 using AutoGPTQ
    
    Args:
        model_name: HuggingFace model identifier
        output_dir: Directory to save quantized model
        bits: Quantization bits (4 or 8)
        group_size: Group size for quantization
    """
    
    print(f"=== INT4 Quantization with GPTQ ===")
    print(f"Model: {model_name}")
    print(f"Output: {output_dir}")
    print(f"Bits: {bits}, Group Size: {group_size}")
    
    try:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    except ImportError:
        print("\n✗ auto-gptq not installed")
        print("Install with: pip install auto-gptq")
        return False
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load model
    print("\n[1/4] Loading model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        quantize_config = BaseQuantizeConfig(
            bits=bits,
            group_size=group_size,
            desc_act=False  # Disable activation quantization for stability
        )
        
        model = AutoGPTQForCausalLM.from_pretrained(
            model_name,
            quantize_config=quantize_config
        )
        
        print("✓ Model loaded")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False
    
    # Step 2: Prepare calibration data
    print("\n[2/4] Preparing calibration data...")
    
    calibration_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "In a fantasy world, a brave knight embarks on a quest.",
        "The wizard cast a powerful spell to protect the village.",
        "Deep in the dungeon, treasure awaits those who dare.",
        "A mysterious stranger arrived at the tavern at midnight."
    ]
    
    calibration_data = [
        tokenizer(text, return_tensors="pt").input_ids
        for text in calibration_texts
    ]
    
    print(f"✓ Prepared {len(calibration_data)} calibration samples")
    
    # Step 3: Quantize
    print("\n[3/4] Quantizing model...")
    try:
        model.quantize(calibration_data)
        print("✓ Quantization complete")
    except Exception as e:
        print(f"✗ Quantization failed: {e}")
        return False
    
    # Step 4: Save
    print("\n[4/4] Saving quantized model...")
    try:
        model.save_quantized(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        print(f"✓ Model saved to {output_dir}")
        
        # Check file sizes
        import os
        total_size = sum(
            os.path.getsize(os.path.join(output_dir, f))
            for f in os.listdir(output_dir)
            if os.path.isfile(os.path.join(output_dir, f))
        )
        
        print(f"  Total size: {total_size / (1024**3):.2f} GB")
        print(f"  Expected speedup: ~1.5x")
        print(f"  Expected memory reduction: ~2x")
        
    except Exception as e:
        print(f"✗ Save failed: {e}")
        return False
    
    print("\n=== Quantization Complete ===")
    return True

def quantize_to_int4_awq(
    model_name: str = "microsoft/Phi-3-mini-4k-instruct",
    output_dir: str = "models/phi3-int4-awq"
):
    """
    Quantize model to INT4 using AWQ (Activation-aware Weight Quantization)
    """
    
    print(f"=== INT4 Quantization with AWQ ===")
    print(f"Model: {model_name}")
    print(f"Output: {output_dir}")
    
    try:
        from awq import AutoAWQForCausalLM
    except ImportError:
        print("\n✗ autoawq not installed")
        print("Install with: pip install autoawq")
        return False
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("\n[1/3] Loading model...")
    try:
        model = AutoAWQForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("✓ Model loaded")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False
    
    print("\n[2/3] Quantizing...")
    try:
        quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4}
        model.quantize(tokenizer, quant_config=quant_config)
        print("✓ Quantization complete")
    except Exception as e:
        print(f"✗ Quantization failed: {e}")
        return False
    
    print("\n[3/3] Saving...")
    try:
        model.save_quantized(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"✓ Model saved to {output_dir}")
    except Exception as e:
        print(f"✗ Save failed: {e}")
        return False
    
    print("\n=== Quantization Complete ===")
    return True

def main():
    parser = argparse.ArgumentParser(description="Quantize model to INT4")
    parser.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct", help="Model name")
    parser.add_argument("--output", default="models/phi3-int4", help="Output directory")
    parser.add_argument("--method", choices=["gptq", "awq"], default="gptq", help="Quantization method")
    parser.add_argument("--bits", type=int, default=4, help="Quantization bits")
    
    args = parser.parse_args()
    
    if args.method == "gptq":
        quantize_to_int4_gptq(args.model, args.output, args.bits)
    else:
        quantize_to_int4_awq(args.model, args.output)

if __name__ == "__main__":
    main()

"""
export_to_onnx.py - Export Phi-3 model with LoRA adapter to ONNX format

This script exports the fine-tuned NPC model to ONNX for C++ inference.
"""

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse
import sys
from pathlib import Path
import os

try:
    from onnxruntime.quantization import quantize_dynamic, QuantType
except ImportError:
    quantize_dynamic = None

# Add core to path
sys.path.append(str(Path(__file__).parent.parent))
from core.config import BASE_MODEL, ADAPTER_PATH


def export_model_to_onnx(
    base_model_name: str,
    adapter_path: str,
    output_path: str,
    adapter_path: str,
    output_path: str,
    use_merged: bool = True,
    quantize: bool = False
):
    """
    Export model to ONNX format
    
    Args:
        base_model_name: HuggingFace model name (e.g., "microsoft/Phi-3-mini-4k-instruct")
        adapter_path: Path to LoRA adapter weights
        output_path: Output path for .onnx file
        use_merged: Whether to merge adapter into base model before export
    """
    print(f"Loading base model: {base_model_name}")
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="cpu",  # Export on CPU for compatibility
        trust_remote_code=True
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    
    # Load adapter if provided
    if adapter_path and Path(adapter_path).exists():
        print(f"Loading adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        
        if use_merged:
            print("Merging adapter weights into base model...")
            model = model.merge_and_unload()
    else:
        print("No adapter specified, using base model only")
        model = base_model
    
    model.eval()
    
    # Prepare dummy input for ONNX export
    dummy_text = "System: You are a helpful NPC.\nName: NPC\nContext: Test\n\nQuestion: Hello\nAnswer:"
    dummy_input = tokenizer(dummy_text, return_tensors="pt")
    
    print(f"Exporting to ONNX: {output_path}")
    print("This may take several minutes...")
    
    # Export to ONNX
    torch.onnx.export(
        model,
        (dummy_input["input_ids"],),
        output_path,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "sequence"}
        },
        opset_version=14,
        do_constant_folding=True
    )
    
    print(f"✓ Model exported successfully to {output_path}")
    
    if quantize:
        if quantize_dynamic is None:
            print("! Warning: onnxruntime not installed or quantization not available. Skipping quantization.")
        else:
            print("Quantizing model (INT8)...")
            quantized_path = output_path.replace(".onnx", "_int8.onnx")
            
            quantize_dynamic(
                model_input=output_path,
                model_output=quantized_path,
                weight_type=QuantType.QUInt8
            )
            print(f"✓ Quantized model saved to {quantized_path}")
            
            # Optionally replace original with quantized?
            # For now, keep both.
    print(f"Model size: {Path(output_path).stat().st_size / (1024*1024):.2f} MB")


def export_tokenizer(tokenizer_name: str, output_dir: str):
    """Export tokenizer files for C++ usage"""
    print(f"Exporting tokenizer to: {output_dir}")
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    
    # Save tokenizer files
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    tokenizer.save_pretrained(output_dir)
    
    print(f"✓ Tokenizer exported to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Export NPC model to ONNX format")
    parser.add_argument(
        "--base-model",
        default=BASE_MODEL,
        help="Base model name or path"
    )
    parser.add_argument(
        "--adapter",
        default=ADAPTER_PATH,
        help="Path to LoRA adapter"
    )
    parser.add_argument(
        "--output",
        default="onnx_models/npc_model.onnx",
        help="Output path for ONNX model"
    )
    parser.add_argument(
        "--export-tokenizer",
        action="store_true",
        help="Also export tokenizer files"
    )
    parser.add_argument(
        "--tokenizer-output",
        default="onnx_models/tokenizer",
        help="Output directory for tokenizer files"
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Don't merge adapter (export separate)"
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Quantize model to INT8 after export"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Export model
    export_model_to_onnx(
        base_model_name=args.base_model,
        adapter_path=args.adapter,
        output_path=args.output,
        use_merged=not args.no_merge,
        quantize=args.quantize
    )
    
    # Export tokenizer if requested
    if args.export_tokenizer:
        export_tokenizer(args.base_model, args.tokenizer_output)
    
    print("\n=== Export Complete ===")
    print(f"ONNX Model: {args.output}")
    if args.export_tokenizer:
        print(f"Tokenizer: {args.tokenizer_output}")
    
    print("\nNext steps:")
    print("1. Download ONNX Runtime from: https://github.com/microsoft/onnxruntime/releases")
    print("2. Set ONNXRUNTIME_ROOT environment variable")
    print("3. Build C++ project: cd cpp && cmake -B build && cmake --build build")
    print("4. Run: ./cpp/build/npc_cli path/to/model.onnx")


if __name__ == "__main__":
    main()

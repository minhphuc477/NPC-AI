#!/usr/bin/env python3
"""
Export Phi-3-Vision to ONNX format
Converts Microsoft's Phi-3-Vision model to ONNX for C++ inference
"""

import torch
import onnx
from transformers import AutoProcessor, AutoModelForVision2Seq
from onnxruntime.quantization import quantize_dynamic, QuantType
import argparse
import os

def export_phi3_vision_to_onnx(
    model_name: str = "microsoft/Phi-3-vision-128k-instruct",
    output_dir: str = "models",
    quantize: bool = True
):
    """
    Export Phi-3-Vision model to ONNX format
    
    Args:
        model_name: HuggingFace model identifier
        output_dir: Directory to save ONNX models
        quantize: Whether to apply INT8 quantization
    """
    
    print(f"=== Exporting Phi-3-Vision to ONNX ===")
    print(f"Model: {model_name}")
    print(f"Output: {output_dir}")
    print(f"Quantize: {quantize}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load model and processor
    print("\n[1/5] Loading model and processor...")
    try:
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use FP32 for ONNX export
            trust_remote_code=True
        )
        model.eval()
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        print("\nNote: This requires ~8GB RAM and HuggingFace access token")
        print("Alternative: Use smaller vision model like CLIP ViT-B/32")
        return False
    
    # Step 2: Create dummy inputs
    print("\n[2/5] Creating dummy inputs...")
    
    # Vision encoder expects 336x336 images
    dummy_image = torch.randn(1, 3, 336, 336)
    
    # Create dummy inputs for vision encoder
    dummy_inputs = {
        "pixel_values": dummy_image
    }
    
    print(f"  Input shape: {dummy_image.shape}")
    
    # Step 3: Export vision encoder
    print("\n[3/5] Exporting vision encoder to ONNX...")
    
    vision_encoder_path = os.path.join(output_dir, "vision_encoder.onnx")
    
    try:
        # Extract vision model (encoder only)
        vision_model = model.model.vision_model if hasattr(model.model, 'vision_model') else model.vision_model
        
        torch.onnx.export(
            vision_model,
            dummy_image,
            vision_encoder_path,
            input_names=['pixel_values'],
            output_names=['image_embeds'],
            dynamic_axes={
                'pixel_values': {0: 'batch'},
                'image_embeds': {0: 'batch'}
            },
            opset_version=14,
            do_constant_folding=True,
            verbose=False
        )
        
        print(f"✓ Vision encoder exported to {vision_encoder_path}")
        
        # Verify ONNX model
        onnx_model = onnx.load(vision_encoder_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model verified")
        
    except Exception as e:
        print(f"✗ Export failed: {e}")
        print("\nTrying alternative approach with CLIP...")
        return export_clip_vision_encoder(output_dir, quantize)
    
    # Step 4: Quantize model (optional)
    if quantize:
        print("\n[4/5] Quantizing model to INT8...")
        
        quantized_path = os.path.join(output_dir, "vision_encoder_int8.onnx")
        
        try:
            quantize_dynamic(
                vision_encoder_path,
                quantized_path,
                weight_type=QuantType.QUInt8
            )
            
            # Check file sizes
            original_size = os.path.getsize(vision_encoder_path) / (1024 * 1024)
            quantized_size = os.path.getsize(quantized_path) / (1024 * 1024)
            
            print(f"✓ Quantized model saved to {quantized_path}")
            print(f"  Original size: {original_size:.1f} MB")
            print(f"  Quantized size: {quantized_size:.1f} MB")
            print(f"  Compression: {(1 - quantized_size/original_size)*100:.1f}%")
            
        except Exception as e:
            print(f"✗ Quantization failed: {e}")
            print("  Using original FP32 model")
    
    # Step 5: Test inference
    print("\n[5/5] Testing ONNX inference...")
    
    try:
        import onnxruntime as ort
        
        # Load ONNX model
        model_path = quantized_path if quantize and os.path.exists(quantized_path) else vision_encoder_path
        session = ort.InferenceSession(model_path)
        
        # Run inference
        test_input = torch.randn(1, 3, 336, 336).numpy()
        outputs = session.run(None, {'pixel_values': test_input})
        
        print(f"✓ Inference successful")
        print(f"  Output shape: {outputs[0].shape}")
        print(f"  Output dtype: {outputs[0].dtype}")
        
    except Exception as e:
        print(f"⚠ Inference test failed: {e}")
        print("  Model exported but not tested")
    
    print("\n=== Export Complete ===")
    print(f"\nTo use in C++:")
    print(f'  VisionLoader loader;')
    print(f'  loader.Load("{model_path}");')
    
    return True

def export_clip_vision_encoder(output_dir: str, quantize: bool = True):
    """
    Fallback: Export CLIP vision encoder (smaller, faster)
    """
    print("\n=== Exporting CLIP ViT-B/32 (Fallback) ===")
    
    try:
        from transformers import CLIPVisionModel
        
        print("[1/3] Loading CLIP vision model...")
        model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        model.eval()
        
        print("[2/3] Exporting to ONNX...")
        dummy_input = torch.randn(1, 3, 224, 224)  # CLIP uses 224x224
        
        vision_encoder_path = os.path.join(output_dir, "clip_vision_encoder.onnx")
        
        torch.onnx.export(
            model,
            dummy_input,
            vision_encoder_path,
            input_names=['pixel_values'],
            output_names=['image_embeds'],
            dynamic_axes={'pixel_values': {0: 'batch'}},
            opset_version=14
        )
        
        print(f"✓ CLIP vision encoder exported to {vision_encoder_path}")
        
        if quantize:
            print("[3/3] Quantizing...")
            quantized_path = os.path.join(output_dir, "clip_vision_encoder_int8.onnx")
            quantize_dynamic(vision_encoder_path, quantized_path, weight_type=QuantType.QUInt8)
            print(f"✓ Quantized model saved to {quantized_path}")
        
        print("\n✓ CLIP export successful!")
        print("Note: CLIP uses 224x224 input (not 336x336)")
        
        return True
        
    except Exception as e:
        print(f"✗ CLIP export failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Export Phi-3-Vision to ONNX")
    parser.add_argument("--model", default="microsoft/Phi-3-vision-128k-instruct", help="Model name")
    parser.add_argument("--output", default="models", help="Output directory")
    parser.add_argument("--no-quantize", action="store_true", help="Skip INT8 quantization")
    parser.add_argument("--clip-fallback", action="store_true", help="Use CLIP instead of Phi-3-Vision")
    
    args = parser.parse_args()
    
    if args.clip_fallback:
        export_clip_vision_encoder(args.output, not args.no_quantize)
    else:
        export_phi3_vision_to_onnx(args.model, args.output, not args.no_quantize)

if __name__ == "__main__":
    main()
